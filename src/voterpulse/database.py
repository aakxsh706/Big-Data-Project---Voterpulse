"""SQLite storage: connections, schema migrations and parameterised queries.

Concurrency model: the scheduler/pipeline process is the only writer; the Streamlit
dashboard only reads. WAL journal mode lets readers continue while a write
transaction is open, and ``busy_timeout`` makes a second writer (e.g. a manual
``run_pipeline.py`` while the scheduler runs) wait instead of failing.

Schema changes are applied as numbered migrations tracked in ``PRAGMA user_version``.
To change the schema, append a new entry to ``MIGRATIONS`` — never edit old ones.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from voterpulse.logging_utils import fields, get_logger
from voterpulse.models import (
    ClassificationOutcome,
    PreparedArticle,
    SentimentScores,
    to_utc_iso,
    utc_now,
)

log = get_logger(__name__)

MIGRATIONS: list[str] = [
    # ---- 1: initial schema -------------------------------------------------------
    """
    CREATE TABLE meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE articles (
        id                  INTEGER PRIMARY KEY,
        headline            TEXT NOT NULL,
        normalized_headline TEXT NOT NULL,
        headline_hash       TEXT NOT NULL UNIQUE,
        source              TEXT,
        url                 TEXT,
        published_at        TEXT,              -- UTC ISO-8601, NULL if feed gave none
        ingested_at         TEXT NOT NULL,     -- UTC ISO-8601
        feed_key            TEXT,              -- first feed that returned this headline
        is_synthetic        INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1))
    );
    CREATE INDEX idx_articles_published ON articles(published_at);
    CREATE INDEX idx_articles_ingested ON articles(ingested_at);

    CREATE TABLE classifications (
        id                INTEGER PRIMARY KEY,
        article_id        INTEGER NOT NULL UNIQUE REFERENCES articles(id) ON DELETE CASCADE,
        primary_issue     TEXT NOT NULL,
        secondary_issue   TEXT,
        evidence          TEXT,
        method            TEXT NOT NULL CHECK (method IN ('llm', 'keyword_fallback')),
        keyword_issue     TEXT NOT NULL,
        model_id          TEXT,
        prompt_version    TEXT,
        raw_response      TEXT,
        validation_status TEXT NOT NULL CHECK (validation_status IN
                          ('valid', 'evidence_unverified', 'llm_invalid', 'llm_unavailable')),
        error             TEXT,
        latency_ms        REAL,
        from_cache        INTEGER NOT NULL DEFAULT 0,
        is_relevant       INTEGER NOT NULL CHECK (is_relevant IN (0, 1)),
        classified_at     TEXT NOT NULL,
        CHECK (secondary_issue IS NULL OR secondary_issue <> primary_issue)
    );
    CREATE INDEX idx_class_issue ON classifications(primary_issue, is_relevant);
    CREATE INDEX idx_class_method ON classifications(method);

    CREATE TABLE classification_cache (
        headline_hash     TEXT NOT NULL,
        model_id          TEXT NOT NULL,
        prompt_version    TEXT NOT NULL,
        primary_issue     TEXT NOT NULL,
        secondary_issue   TEXT,
        evidence          TEXT,
        raw_response      TEXT,
        validation_status TEXT NOT NULL,
        created_at        TEXT NOT NULL,
        PRIMARY KEY (headline_hash, model_id, prompt_version)
    );

    CREATE TABLE sentiment_scores (
        article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
        positive   REAL NOT NULL,
        negative   REAL NOT NULL,
        neutral    REAL NOT NULL,
        compound   REAL NOT NULL CHECK (compound BETWEEN -1 AND 1),
        label      TEXT NOT NULL CHECK (label IN ('Positive', 'Neutral', 'Negative')),
        scored_at  TEXT NOT NULL
    );
    CREATE INDEX idx_sentiment_label ON sentiment_scores(label);

    CREATE TABLE review_queue (
        id              INTEGER PRIMARY KEY,
        article_id      INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
        llm_label       TEXT,
        keyword_label   TEXT,
        reason          TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN
                        ('pending', 'llm_correct', 'keyword_correct', 'both_wrong', 'dismissed')),
        corrected_issue TEXT,
        reviewer_notes  TEXT,
        created_at      TEXT NOT NULL,
        reviewed_at     TEXT,
        UNIQUE (article_id, reason)
    );
    CREATE INDEX idx_review_status ON review_queue(status);

    CREATE TABLE pipeline_runs (
        id                     INTEGER PRIMARY KEY,
        started_at             TEXT NOT NULL,
        completed_at           TEXT,
        status                 TEXT NOT NULL CHECK (status IN
                               ('running', 'success', 'partial', 'failed', 'abandoned')),
        trigger                TEXT,
        headlines_collected    INTEGER DEFAULT 0,
        new_headlines          INTEGER DEFAULT 0,
        duplicates             INTEGER DEFAULT 0,
        classified             INTEGER DEFAULT 0,
        llm_classified         INTEGER DEFAULT 0,
        fallback_classified    INTEGER DEFAULT 0,
        rejected               INTEGER DEFAULT 0,
        failed                 INTEGER DEFAULT 0,
        feeds_ok               INTEGER DEFAULT 0,
        feeds_failed           INTEGER DEFAULT 0,
        avg_classification_ms  REAL,
        duration_seconds       REAL,
        error_info             TEXT
    );
    CREATE INDEX idx_runs_started ON pipeline_runs(started_at);

    CREATE TABLE alerts (
        id                 INTEGER PRIMARY KEY,
        issue              TEXT NOT NULL,
        direction          TEXT NOT NULL CHECK (direction IN ('more_positive', 'more_negative')),
        severity           TEXT NOT NULL CHECK (severity IN ('noticeable', 'marked')),
        previous_sentiment REAL NOT NULL,
        current_sentiment  REAL NOT NULL,
        magnitude          REAL NOT NULL,
        current_count      INTEGER NOT NULL,
        reference_count    INTEGER NOT NULL,
        message            TEXT NOT NULL,
        triggered_at       TEXT NOT NULL,
        last_checked_at    TEXT NOT NULL,
        status             TEXT NOT NULL CHECK (status IN ('active', 'resolved')),
        resolved_at        TEXT
    );
    -- At most one active alert per issue: a continuing shift updates that row.
    CREATE UNIQUE INDEX idx_alerts_one_active ON alerts(issue) WHERE status = 'active';
    """,
]


class DatasetKindError(RuntimeError):
    """Raised when synthetic (demo) and live data would be mixed in one database."""


def connect(db_path: Path, busy_timeout_ms: int = 10000) -> sqlite3.Connection:
    """Open a connection with foreign keys, WAL and a busy timeout enabled.

    The connection is in autocommit mode; use :func:`transaction` for atomic writes.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=busy_timeout_ms / 1000, isolation_level=None,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection, immediate: bool = True) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, rolling back on any exception."""
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order. Returns the resulting schema version."""
    current = schema_version(conn)
    for version, script in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        # executescript commits implicitly, so wrap the script itself in a transaction.
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;")
        log.info("schema_migrated", extra=fields(version=version))
    return schema_version(conn)


def init_db(db_path: Path, kind: str = "live", busy_timeout_ms: int = 10000) -> sqlite3.Connection:
    """Create/upgrade the database and stamp it as ``live`` or ``demo``.

    Raises :class:`DatasetKindError` if the file was created for the other kind, so
    synthetic data can never be written into the live database or vice versa.
    """
    if kind not in {"live", "demo"}:
        raise ValueError("kind must be 'live' or 'demo'")
    conn = connect(db_path, busy_timeout_ms)
    migrate(conn)
    row = conn.execute("SELECT value FROM meta WHERE key = 'dataset_kind'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta(key, value) VALUES ('dataset_kind', ?)", (kind,))
    elif row["value"] != kind:
        conn.close()
        raise DatasetKindError(f"{db_path} holds {row['value']!r} data; refusing to use it as {kind!r}")
    return conn


def dataset_kind(conn: sqlite3.Connection) -> str | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'dataset_kind'").fetchone()
    except sqlite3.OperationalError:
        return None
    return row["value"] if row else None


# --------------------------------------------------------------------------- articles
def insert_article(conn: sqlite3.Connection, article: PreparedArticle,
                   ingested_at: datetime | None = None, is_synthetic: bool = False) -> int | None:
    """Insert an article; returns its id, or ``None`` if the headline hash already exists."""
    if is_synthetic and dataset_kind(conn) != "demo":
        raise DatasetKindError("synthetic articles may only be written to the demo database")
    cur = conn.execute(
        """INSERT OR IGNORE INTO articles
           (headline, normalized_headline, headline_hash, source, url, published_at,
            ingested_at, feed_key, is_synthetic)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (article.headline, article.normalized_headline, article.headline_hash, article.source,
         article.url, to_utc_iso(article.published_at), to_utc_iso(ingested_at or utc_now()),
         article.feed_key, int(is_synthetic)),
    )
    return int(cur.lastrowid) if cur.rowcount == 1 else None


def existing_hashes(conn: sqlite3.Connection, hashes: list[str]) -> set[str]:
    """Return which of ``hashes`` are already stored (chunked to respect SQLite limits)."""
    found: set[str] = set()
    for start in range(0, len(hashes), 500):
        chunk = hashes[start:start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT headline_hash FROM articles WHERE headline_hash IN ({placeholders})", chunk
        ).fetchall()
        found.update(r["headline_hash"] for r in rows)
    return found


def unclassified_articles(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Articles without a classification, oldest ingestion first."""
    return conn.execute(
        """SELECT a.* FROM articles a
           LEFT JOIN classifications c ON c.article_id = a.id
           WHERE c.id IS NULL
           ORDER BY a.ingested_at, a.id
           LIMIT ?""",
        (limit,),
    ).fetchall()


def count_unclassified(conn: sqlite3.Connection) -> int:
    return int(conn.execute(
        """SELECT COUNT(*) FROM articles a LEFT JOIN classifications c ON c.article_id = a.id
           WHERE c.id IS NULL"""
    ).fetchone()[0])


def relevant_unscored_articles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Accepted (relevant) articles that still lack a sentiment score."""
    return conn.execute(
        """SELECT a.id, a.headline FROM articles a
           JOIN classifications c ON c.article_id = a.id AND c.is_relevant = 1
           LEFT JOIN sentiment_scores s ON s.article_id = a.id
           WHERE s.article_id IS NULL"""
    ).fetchall()


# --------------------------------------------------------------------------- classification
def get_cached_classification(conn: sqlite3.Connection, headline_hash: str, model_id: str,
                              prompt_version: str) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT * FROM classification_cache
           WHERE headline_hash = ? AND model_id = ? AND prompt_version = ?""",
        (headline_hash, model_id, prompt_version),
    ).fetchone()


def put_cached_classification(conn: sqlite3.Connection, headline_hash: str,
                              outcome: ClassificationOutcome) -> None:
    """Cache a validated LLM result (fallback results are never cached)."""
    if outcome.method.value != "llm" or not outcome.model_id or not outcome.prompt_version:
        return
    conn.execute(
        """INSERT OR REPLACE INTO classification_cache
           (headline_hash, model_id, prompt_version, primary_issue, secondary_issue, evidence,
            raw_response, validation_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (headline_hash, outcome.model_id, outcome.prompt_version, outcome.primary_issue,
         outcome.secondary_issue, outcome.evidence, outcome.raw_response,
         outcome.validation_status.value, to_utc_iso(utc_now())),
    )


def save_classification(conn: sqlite3.Connection, article_id: int, outcome: ClassificationOutcome,
                        is_relevant: bool, classified_at: datetime | None = None) -> None:
    conn.execute(
        """INSERT INTO classifications
           (article_id, primary_issue, secondary_issue, evidence, method, keyword_issue,
            model_id, prompt_version, raw_response, validation_status, error, latency_ms,
            from_cache, is_relevant, classified_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (article_id, outcome.primary_issue, outcome.secondary_issue, outcome.evidence,
         outcome.method.value, outcome.keyword_issue, outcome.model_id, outcome.prompt_version,
         outcome.raw_response, outcome.validation_status.value, outcome.error,
         outcome.latency_ms, int(outcome.from_cache), int(is_relevant),
         to_utc_iso(classified_at or utc_now())),
    )


def add_review_item(conn: sqlite3.Connection, article_id: int, llm_label: str | None,
                    keyword_label: str | None, reason: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO review_queue
           (article_id, llm_label, keyword_label, reason, status, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?)""",
        (article_id, llm_label, keyword_label, reason, to_utc_iso(utc_now())),
    )


def update_review_item(conn: sqlite3.Connection, review_id: int, status: str,
                       reviewer_notes: str | None = None, corrected_issue: str | None = None) -> None:
    """Record a human review decision. Does not alter the stored classification."""
    conn.execute(
        """UPDATE review_queue SET status = ?, reviewer_notes = ?, corrected_issue = ?,
                  reviewed_at = ? WHERE id = ?""",
        (status, reviewer_notes, corrected_issue, to_utc_iso(utc_now()), review_id),
    )


# --------------------------------------------------------------------------- sentiment
def save_sentiment(conn: sqlite3.Connection, article_id: int, scores: SentimentScores,
                   scored_at: datetime | None = None) -> bool:
    """Store one score per article. Returns False if the article was already scored."""
    cur = conn.execute(
        """INSERT INTO sentiment_scores
           (article_id, positive, negative, neutral, compound, label, scored_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(article_id) DO NOTHING""",
        (article_id, scores.positive, scores.negative, scores.neutral, scores.compound,
         scores.label.value, to_utc_iso(scored_at or utc_now())),
    )
    return cur.rowcount == 1


# --------------------------------------------------------------------------- pipeline runs
def start_run(conn: sqlite3.Connection, trigger: str, stale_after_minutes: int) -> int | None:
    """Atomically claim the pipeline. Returns the new run id, or None if another run is active.

    A run left in ``running`` for longer than ``stale_after_minutes`` (process crash) is
    marked ``abandoned`` so it cannot block future runs forever.
    """
    now = utc_now()
    cutoff = to_utc_iso(now - timedelta(minutes=stale_after_minutes))
    with transaction(conn):
        conn.execute(
            """UPDATE pipeline_runs SET status = 'abandoned', completed_at = ?,
                      error_info = COALESCE(error_info, 'run did not finish (stale lock)')
               WHERE status = 'running' AND started_at < ?""",
            (to_utc_iso(now), cutoff),
        )
        if conn.execute("SELECT 1 FROM pipeline_runs WHERE status = 'running'").fetchone():
            return None
        cur = conn.execute(
            "INSERT INTO pipeline_runs (started_at, status, trigger) VALUES (?, 'running', ?)",
            (to_utc_iso(now), trigger),
        )
        return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, stats: dict[str, Any]) -> None:
    allowed = {"headlines_collected", "new_headlines", "duplicates", "classified",
               "llm_classified", "fallback_classified", "rejected", "failed", "feeds_ok",
               "feeds_failed", "avg_classification_ms", "duration_seconds", "error_info"}
    unknown = set(stats) - allowed
    if unknown:
        raise ValueError(f"unknown run statistics: {sorted(unknown)}")
    columns = ", ".join(f"{key} = ?" for key in stats)
    params = list(stats.values()) + [status, to_utc_iso(utc_now()), run_id]
    sep = ", " if columns else ""
    conn.execute(
        f"UPDATE pipeline_runs SET {columns}{sep}status = ?, completed_at = ? WHERE id = ?",
        params,
    )


# --------------------------------------------------------------------------- alerts
def active_alert(conn: sqlite3.Connection, issue: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM alerts WHERE issue = ? AND status = 'active'", (issue,)
    ).fetchone()
