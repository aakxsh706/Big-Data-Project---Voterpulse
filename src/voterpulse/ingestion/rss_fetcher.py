"""Google News RSS fetching with timeouts, bounded retries and per-feed isolation.

A failure in one feed is recorded in its :class:`FeedResult` and never raised, so
the remaining feeds are still collected.
"""

from __future__ import annotations

import calendar
import time
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlencode

import feedparser
import requests

from voterpulse.config import FeedDefinition, RssConfig
from voterpulse.logging_utils import fields, get_logger
from voterpulse.models import FeedResult, RawHeadline

log = get_logger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def build_feed_url(cfg: RssConfig, feed: FeedDefinition) -> str:
    return f"{cfg.base_url}?{urlencode({'q': feed.query, **cfg.params})}"


def _entry_datetime(entry: feedparser.FeedParserDict) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                # feedparser returns a UTC struct_time; timegm interprets it as UTC.
                return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
    return None


def _entry_source(entry: feedparser.FeedParserDict) -> str | None:
    source = entry.get("source")
    if isinstance(source, dict) and source.get("title"):
        return str(source["title"]).strip()
    title = entry.get("title", "")
    # Fallback: Google News titles end with " - Publisher".
    if " - " in title:
        return title.rsplit(" - ", 1)[1].strip() or None
    return None


def parse_feed(content: bytes, feed_key: str) -> tuple[list[RawHeadline], str | None]:
    """Parse RSS bytes. Returns ``(headlines, error)``; error is set for unusable feeds."""
    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        return [], f"malformed RSS: {parsed.get('bozo_exception')!r}"
    headlines: list[RawHeadline] = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        headlines.append(RawHeadline(
            headline=title,
            url=entry.get("link") or None,
            source=_entry_source(entry),
            published_at=_entry_datetime(entry),
            feed_key=feed_key,
        ))
    return headlines, None


def fetch_feed(cfg: RssConfig, feed: FeedDefinition, session: requests.Session | None = None,
               sleep: Callable[[float], None] = time.sleep) -> FeedResult:
    """Fetch and parse one feed with bounded exponential backoff on transient errors."""
    session = session or requests.Session()
    url = build_feed_url(cfg, feed)
    req = cfg.request
    last_error = "unknown error"
    status: int | None = None
    for attempt in range(req.max_retries + 1):
        try:
            response = session.get(url, timeout=req.timeout_seconds,
                                   headers={"User-Agent": req.user_agent})
            status = response.status_code
            if status in RETRYABLE_STATUS:
                last_error = f"HTTP {status}"
            elif status >= 400:
                log.warning("feed_failed", extra=fields(feed=feed.key, status=status))
                return FeedResult(feed.key, False, error=f"HTTP {status}", status_code=status)
            else:
                headlines, error = parse_feed(response.content, feed.key)
                if error:
                    log.warning("feed_malformed", extra=fields(feed=feed.key, error=error))
                    return FeedResult(feed.key, False, error=error, status_code=status)
                log.info("feed_fetched", extra=fields(feed=feed.key, entries=len(headlines)))
                return FeedResult(feed.key, True, headlines=headlines, status_code=status)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except requests.RequestException as exc:  # non-transient request problem
            log.warning("feed_failed", extra=fields(feed=feed.key, error=exc))
            return FeedResult(feed.key, False, error=str(exc), status_code=status)
        if attempt < req.max_retries:
            delay = req.backoff_seconds * (2 ** attempt)
            log.info("feed_retry", extra=fields(feed=feed.key, attempt=attempt + 1,
                                                 delay_s=delay, error=last_error))
            sleep(delay)
    log.warning("feed_failed", extra=fields(feed=feed.key, error=last_error))
    return FeedResult(feed.key, False, error=last_error, status_code=status)


def fetch_all(cfg: RssConfig, session: requests.Session | None = None,
              sleep: Callable[[float], None] = time.sleep) -> list[FeedResult]:
    """Fetch every configured feed sequentially, pausing politely between requests."""
    session = session or requests.Session()
    results: list[FeedResult] = []
    for index, feed in enumerate(cfg.feeds):
        if index:
            sleep(cfg.request.pause_between_feeds_seconds)
        try:
            results.append(fetch_feed(cfg, feed, session, sleep))
        except Exception as exc:  # defensive: never let one feed crash the run
            log.exception("feed_unexpected_error", extra=fields(feed=feed.key))
            results.append(FeedResult(feed.key, False, error=f"unexpected: {exc}"))
    return results
