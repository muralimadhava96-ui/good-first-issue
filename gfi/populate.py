#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from operator import itemgetter
from os import getenv, path
from typing import TypedDict, Dict, Union, List, Optional

import toml

from github3 import exceptions, login
from numerize import numerize
from emoji import emojize
from slugify import slugify
from loguru import logger


MAX_CONCURRENCY = 5
REPO_DATA_FILE = "data/repositories.toml"
REPO_GENERATED_DATA_FILE = "data/generated.json"
TAGS_GENERATED_DATA_FILE = "data/tags.json"
LABELS_DATA_FILE = "data/labels.json"

GH_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/(?P<owner>[\w\.-]+)/(?P<name>[\w\.-]+)/?"
)

ISSUE_STATE = "open"
ISSUE_SORT = "created"
ISSUE_SORT_DIRECTION = "desc"
ISSUE_LIMIT = 10
SLUGIFY_REPLACEMENTS = [["#", "sharp"], ["+", "plus"]]
MAX_INACTIVITY_DAYS = 90


class RepositoryIdentifier(TypedDict):
    owner: str
    name: str


class RepositoryInfo(TypedDict):
    id: str
    name: str
    owner: str
    description: str
    language: str
    slug: str
    url: str
    stars: int
    stars_display: str
    last_modified: str
    issues: List[Dict[str, Union[str, int]]]


class GitHubRateLimiter:
    """Thread-safe rate limiter for GitHub API requests."""

    def __init__(self, client, requests_per_second: float = 1.0):
        self._client = client
        self._lock = threading.Lock()
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._remaining: Optional[int] = None
        self._reset_time: Optional[float] = None
        self._paused_until = 0.0

    def acquire(self):
        wait_time = 0.0

        with self._lock:
            now = time.time()

            if now < self._paused_until:
                wait_time = self._paused_until - now

            if self._remaining is None or (self._remaining % 100 == 0):
                self._update_rate_limit()

            if self._remaining is not None and self._remaining < 100:
                if self._reset_time:
                    wait_time = max(wait_time, self._reset_time - now + 5)

            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = max(wait_time, self._min_interval - elapsed)

            self._last_request_time = time.time()

            if self._remaining is not None:
                self._remaining -= 1

        if wait_time > 0:
            logger.warning("Waiting {:.0f}s due to rate limiting", wait_time)
            time.sleep(wait_time)

    def _update_rate_limit(self):
        try:
            info = self._client.rate_limit()["resources"]["core"]
            self._remaining = info["remaining"]
            self._reset_time = info["reset"]
            logger.debug("Rate limit: {}/{}", self._remaining, info["limit"])
        except Exception as e:
            logger.warning("Failed to check rate limit: {}", e)

    def report_rate_limit_hit(self):
        with self._lock:
            self._update_rate_limit()
            if self._reset_time:
                wait_time = max(60, self._reset_time - time.time() + 5)
            else:
                wait_time = 60
            self._paused_until = time.time() + wait_time
            self._remaining = 0
            logger.warning("Rate limit hit. Pausing all workers for {:.0f}s", wait_time)


def parse_github_url(url: str) -> Optional[RepositoryIdentifier]:
    match = GH_URL_PATTERN.search(url)
    if match:
        return match.groupdict()
    return None


def get_repository_info(
    identifier: RepositoryIdentifier,
    client,
    rate_limiter: GitHubRateLimiter,
    issue_labels: List[str],
) -> Optional[RepositoryInfo]:

    owner, name = identifier["owner"], identifier["name"]
    logger.info("Getting info for {}/{}", owner, name)

    max_retries = 3

    for attempt in range(max_retries):
        try:
            rate_limiter.acquire()
            repository = client.repository(owner, name)

            if repository.archived:
                return None

            if not repository.pushed_at:
                return None

            days_since_push = (
                datetime.now(timezone.utc) - repository.pushed_at
            ).days

            if days_since_push > MAX_INACTIVITY_DAYS:
                logger.info(
                    "Skipping {}/{} due to inactivity ({} days)",
                    owner,
                    name,
                    days_since_push,
                )
                return None

            good_first_issues = set()

            for label in issue_labels:
                rate_limiter.acquire()
                issues_for_label = list(
                    repository.issues(
                        labels=label,
                        state=ISSUE_STATE,
                        number=ISSUE_LIMIT,
                        sort=ISSUE_SORT,
                        direction=ISSUE_SORT_DIRECTION,
                    )
                )
                good_first_issues.update(issues_for_label)

            if not good_first_issues or not repository.language:
                return None

            info: RepositoryInfo = {
                "id": str(repository.id),
                "name": name,
                "owner": owner,
                "description": emojize(repository.description or ""),
                "language": repository.language,
                "slug": slugify(repository.language, replacements=SLUGIFY_REPLACEMENTS),
                "url": repository.html_url,
                "stars": repository.stargazers_count,
                "stars_display": numerize.numerize(repository.stargazers_count),
                "last_modified": repository.pushed_at.isoformat(),
                "issues": [
                    {
                        "title": issue.title,
                        "url": issue.html_url,
                        "number": issue.number,
                        "comments_count": issue.comments_count,
                        "created_at": issue.created_at.isoformat(),
                    }
                    for issue in good_first_issues
                ],
            }

            return info

        except exceptions.ForbiddenError:
            rate_limiter.report_rate_limit_hit()
            if attempt == max_retries - 1:
                logger.error("Rate limit exceeded for {}/{}", owner, name)

        except (exceptions.NotFoundError, exceptions.ConnectionError):
            logger.warning("Failed to fetch {}/{}", owner, name)
            return None

    return None


if __name__ == "__main__":

    if not path.exists(REPO_DATA_FILE):
        raise RuntimeError("No config data file found.")

    if not path.exists(LABELS_DATA_FILE):
        raise RuntimeError("No labels data file found.")

    token = getenv("GH_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GH_ACCESS_TOKEN not set.")

    with open(LABELS_DATA_FILE) as labels_file:
        labels_data = json.load(labels_file)
        ISSUE_LABELS = labels_data["labels"]

    DATA = toml.load(REPO_DATA_FILE)

    repositories = list(
        filter(
            None,
            (parse_github_url(url) for url in DATA["repositories"]),
        )
    )

    random.shuffle(repositories)

    client = login(token=token)
    rate_limiter = GitHubRateLimiter(client)

    REPOSITORIES: List[RepositoryInfo] = []
    TAGS: Counter = Counter()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        results = executor.map(
            lambda identifier: get_repository_info(
                identifier, client, rate_limiter, ISSUE_LABELS
            ),
            repositories,
        )

        for result in results:
            if result:
                REPOSITORIES.append(result)
                TAGS[result["language"]] += 1

    with open(REPO_GENERATED_DATA_FILE, "w") as f:
        json.dump(REPOSITORIES, f, indent=2)

    tags = [
        {
            "language": key,
            "count": value,
            "slug": slugify(key, replacements=SLUGIFY_REPLACEMENTS),
        }
        for key, value in TAGS.items()
        if value >= 3
    ]

    tags_sorted = sorted(tags, key=itemgetter("count"), reverse=True)

    with open(TAGS_GENERATED_DATA_FILE, "w") as f:
        json.dump(tags_sorted, f, indent=2)

    logger.info("Generated {} repositories", len(REPOSITORIES))
    logger.info("Generated {} tags", len(tags_sorted))
