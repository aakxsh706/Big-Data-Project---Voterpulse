"""Durable, bounded synchronization of the operational journal to MongoDB.

SQLite triggers queue article changes in the same transaction as the change. Mongo
documents are replaced in full so reclassification cannot leave stale fields behind.
Only successful bulk writes acknowledge queue revisions; retries are idempotent.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timezone
from typing import Any

from voterpulse import database as db
from voterpulse.config import AppConfig
from voterpulse.models import parse_utc_iso


def _pymongo():
    try:
        import pymongo
    except ImportError as exc:
        raise RuntimeError(
            'MongoDB support is optional. Install it with: pip install -e ".[big-data]"'
        ) from exc
    return pymongo


@contextmanager
def open_mongo(cfg: AppConfig, kind: str, *, initialize: bool = False) -> Iterator[Any]:
    """Open a UTC-aware Mongo database; only explicit writers create metadata/indexes."""
    if kind not in {"live", "demo"}:
        raise ValueError("kind must be 'live' or 'demo'")
    settings = cfg.settings.big_data
    if settings.mongo_database == settings.mongo_demo_database:
        raise db.DatasetKindError("MongoDB live and demo databases must have different names")
    name = settings.mongo_demo_database if kind == "demo" else settings.mongo_database
    pymongo = _pymongo()
    client = pymongo.MongoClient(
        settings.mongo_uri.get_secret_value(),
        serverSelectionTimeoutMS=settings.mongo_timeout_ms,
        connectTimeoutMS=settings.mongo_timeout_ms,
        socketTimeoutMS=settings.mongo_timeout_ms,
        w="majority",
        wTimeoutMS=settings.mongo_timeout_ms,
        tz_aware=True,
    )
    try:
        client.admin.command("ping")
        mongo = client[name]
        if initialize:
            mongo.meta.update_one(
                {"_id": "dataset_kind"}, {"$setOnInsert": {"value": kind}}, upsert=True
            )
        stamp = mongo.meta.find_one({"_id": "dataset_kind"})
        if stamp is None or stamp.get("value") != kind:
            raise db.DatasetKindError(f"MongoDB database {name!r} is not stamped as {kind!r}")
        if initialize:
            mongo.articles.create_index([("dataset_kind", 1), ("is_relevant", 1), ("event_time", 1)])
            mongo.articles.create_index([("issue", 1), ("event_time", 1)])
            mongo.spark_runs.create_index([("dataset_kind", 1), ("status", 1), ("completed_at", -1)])
            mongo.daily_summaries.create_index([
                ("run_id", 1), ("dataset_kind", 1), ("day", -1), ("issue", 1)
            ])
        yield mongo
    finally:
        client.close()


_ARTICLE_QUERY = """
SELECT q.article_id AS queued_article_id, q.revision AS sync_revision, a.*,
       c.primary_issue AS issue, c.secondary_issue, c.evidence, c.method,
       c.keyword_issue, c.model_id, c.prompt_version, c.raw_response,
       c.validation_status, c.error AS classification_error,
       c.latency_ms AS classification_latency_ms, c.from_cache,
       c.is_relevant, c.classified_at, c.region_status, c.region_evidence,
       s.positive, s.negative, s.neutral, s.compound, s.label, s.scored_at
FROM article_sync_outbox q
LEFT JOIN articles a ON a.id = q.article_id
LEFT JOIN classifications c ON c.article_id = a.id
LEFT JOIN sentiment_scores s ON s.article_id = a.id
ORDER BY q.revision, q.article_id
LIMIT ?
"""


def _document(row: sqlite3.Row, kind: str) -> dict[str, Any]:
    document = dict(row)
    document.pop("queued_article_id")
    document["article_id"] = document.pop("id")
    document["_id"] = document["headline_hash"]
    document["dataset_kind"] = kind
    document["is_synthetic"] = bool(document["is_synthetic"])
    if document["is_synthetic"] and kind != "demo":
        raise db.DatasetKindError("Synthetic articles cannot be synchronized to live MongoDB")
    document["is_relevant"] = bool(document["is_relevant"])
    document["from_cache"] = bool(document["from_cache"])
    document["region_status"] = document["region_status"] or "not_checked"
    for key in ("published_at", "ingested_at", "classified_at", "scored_at"):
        value = parse_utc_iso(document[key])
        document[key] = value.astimezone(timezone.utc) if value is not None else None
    published, ingested = document["published_at"], document["ingested_at"]
    document["event_time"] = ingested if published is None or published > ingested else published
    return document


def pending_articles(conn: sqlite3.Connection) -> int:
    """Number of pending article revisions and deletion tombstones."""
    return int(conn.execute("SELECT COUNT(*) FROM article_sync_outbox").fetchone()[0]) + int(
        conn.execute("SELECT COUNT(*) FROM article_sync_deletions").fetchone()[0]
    )


def sync_articles(conn: sqlite3.Connection, cfg: AppConfig, kind: str, *, full: bool = False) -> int:
    """Synchronize pending changes; return the number of Mongo write operations.

    ``full=True`` requeues existing articles for an empty/new MongoDB destination.
    It does not prune extra documents restored from an older backup.
    Each batch briefly holds the SQLite write lock while Mongo acknowledges writes,
    serializing concurrent syncs and preventing an older snapshot from winning.
    Network timeouts and batch sizes bound this lock; failures retain the queue.
    """
    if db.dataset_kind(conn) != kind:
        raise db.DatasetKindError("SQLite source and MongoDB destination kinds do not match")
    size = cfg.settings.big_data.sync_batch_size
    if size < 1:
        raise ValueError("sync_batch_size must be positive")
    if full:
        with db.transaction(conn):
            conn.execute("UPDATE article_sync_clock SET revision = revision + 1 WHERE id = 1")
            conn.execute("""
                INSERT INTO article_sync_outbox(article_id, revision)
                SELECT a.id, clock.revision FROM articles a CROSS JOIN article_sync_clock clock
                WHERE clock.id = 1
                ON CONFLICT(article_id) DO UPDATE SET revision = excluded.revision
            """)
    pymongo = _pymongo()
    written = 0
    with open_mongo(cfg, kind, initialize=True) as mongo:
        source = conn.execute("SELECT value FROM meta WHERE key = 'article_sync_source'").fetchone()
        if source is None:
            raise RuntimeError("Article sync source identity is missing; initialize the database migrations first")
        mongo.meta.update_one(
            {"_id": "source_instance"}, {"$setOnInsert": {"value": source[0]}}, upsert=True
        )
        if mongo.meta.find_one({"_id": "source_instance"}).get("value") != source[0]:
            raise db.DatasetKindError(
                "MongoDB belongs to a different SQLite journal. Restore the original SQLite file "
                "or configure a fresh MongoDB destination database before synchronizing."
            )
        while True:
            with db.transaction(conn):
                deleted = conn.execute(
                    "SELECT headline_hash, revision FROM article_sync_deletions ORDER BY revision LIMIT ?",
                    (size,),
                ).fetchall()
                rows = conn.execute(_ARTICLE_QUERY, (size - len(deleted),)).fetchall()
                if not deleted and not rows:
                    break
                operations = [pymongo.DeleteOne({"_id": row["headline_hash"]}) for row in deleted]
                for row in rows:
                    if row["id"] is not None:
                        document = _document(row, kind)
                        operations.append(pymongo.ReplaceOne({"_id": document["_id"]}, document, upsert=True))
                if operations:
                    # Ordering matters when an article was deleted and recreated with the same hash.
                    mongo.articles.bulk_write(operations, ordered=True)
                conn.executemany(
                    "DELETE FROM article_sync_deletions WHERE headline_hash = ? AND revision = ?",
                    [(row["headline_hash"], row["revision"]) for row in deleted],
                )
                conn.executemany(
                    "DELETE FROM article_sync_outbox WHERE article_id = ? AND revision = ?",
                    [(row["queued_article_id"], row["sync_revision"]) for row in rows],
                )
                written += len(operations)
    return written
