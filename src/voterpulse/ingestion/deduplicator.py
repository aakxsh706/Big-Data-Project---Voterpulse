"""Duplicate detection within a fetch batch and against the database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from voterpulse.database import existing_hashes
from voterpulse.ingestion.normalizer import clean_headline, headline_hash, normalize_for_dedup
from voterpulse.models import PreparedArticle, RawHeadline


def prepare(raw: RawHeadline) -> PreparedArticle | None:
    """Clean and hash one raw headline. Returns None for empty/unusable titles."""
    headline = clean_headline(raw.headline, raw.source)
    normalized = normalize_for_dedup(headline)
    if len(normalized) < 3:
        return None
    return PreparedArticle(
        headline=headline,
        normalized_headline=normalized,
        headline_hash=headline_hash(normalized),
        url=raw.url,
        source=raw.source.strip() if raw.source else None,
        published_at=raw.published_at,
        feed_key=raw.feed_key,
    )


def deduplicate_batch(raws: Iterable[RawHeadline]) -> tuple[list[PreparedArticle], int, int]:
    """Prepare headlines and drop in-batch duplicates (the same story from several feeds).

    Returns ``(unique_articles, duplicate_count, unusable_count)``. The first occurrence
    wins, so results are deterministic for a given feed order.
    """
    seen: set[str] = set()
    unique: list[PreparedArticle] = []
    duplicates = unusable = 0
    for raw in raws:
        article = prepare(raw)
        if article is None:
            unusable += 1
            continue
        if article.headline_hash in seen:
            duplicates += 1
            continue
        seen.add(article.headline_hash)
        unique.append(article)
    return unique, duplicates, unusable


def filter_new(conn: sqlite3.Connection, articles: list[PreparedArticle]) -> tuple[list[PreparedArticle], int]:
    """Drop articles whose hash is already stored. Returns ``(new_articles, already_stored)``."""
    stored = existing_hashes(conn, [a.headline_hash for a in articles])
    new = [a for a in articles if a.headline_hash not in stored]
    return new, len(articles) - len(new)
