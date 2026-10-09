"""Read-side data access for the dashboard (business logic only, no Streamlit widgets).

The dashboard never runs the pipeline or a scheduler; it only reads what the separate
pipeline process wrote. If the database file does not exist yet it is created empty
so the app starts cleanly on a fresh install.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from voterpulse import database as db
from voterpulse.config import IST, AppConfig, IssueDefinition
from voterpulse.models import parse_utc_iso
from voterpulse.sentiment import aggregator as agg
from voterpulse.sentiment.translator import MoodReport, build_report


def open_db(cfg: AppConfig) -> sqlite3.Connection:
    """Connection to the active database (demo or live), created empty if missing."""
    kind = "demo" if cfg.settings.demo_enabled else "live"
    path = cfg.db_path
    if not path.exists():
        db.init_db(path, kind, cfg.settings.database.busy_timeout_ms).close()
    conn = db.connect(path, cfg.settings.database.busy_timeout_ms)
    db.migrate(conn)  # harmless no-op when current; upgrades an older file
    return conn


@dataclass
class Snapshot:
    """Everything the voter-facing pages need, read in one go."""

    frame: pd.DataFrame
    last_success: datetime | None
    last_run: dict | None
    coverage_start: datetime | None
    alerts: pd.DataFrame
    dataset_kind: str | None
    total_articles: int
    pending: int
    db_path: str
    read_at: datetime = field(default_factory=lambda: datetime.now(tz=IST))


def load_snapshot(cfg: AppConfig) -> Snapshot:
    with closing(open_db(cfg)) as conn:
        frame = agg.load_relevant_frame(conn)
        success = conn.execute(
            """SELECT MAX(completed_at) FROM pipeline_runs
               WHERE status IN ('success', 'partial') AND feeds_ok > 0"""
        ).fetchone()[0]
        last_run = conn.execute("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1").fetchone()
        alerts = pd.read_sql_query(
            "SELECT * FROM alerts WHERE status = 'active' ORDER BY magnitude DESC", conn)
        kind = db.dataset_kind(conn)
        # Demo data has no pipeline runs; its "last update" is the newest generated headline.
        if success is None and kind == "demo":
            success = conn.execute("SELECT MAX(ingested_at) FROM articles").fetchone()[0]
        return Snapshot(
            frame=frame,
            last_success=parse_utc_iso(success),
            last_run=dict(last_run) if last_run else None,
            coverage_start=agg.coverage_start(conn),
            alerts=alerts,
            dataset_kind=kind,
            total_articles=int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]),
            pending=db.count_unclassified(conn),
            db_path=str(cfg.db_path),
        )


@dataclass
class IssueView:
    issue: IssueDefinition
    summary: agg.IssueSummary
    trend: agg.TrendResult
    report: MoodReport


def issue_view(cfg: AppConfig, snap: Snapshot, issue: IssueDefinition, window: agg.Window,
               now: datetime) -> IssueView:
    a = cfg.settings.aggregation
    t = cfg.thresholds
    summary = agg.summarize(snap.frame, issue.name, window, a.min_headlines)
    trend = agg.weekly_trend(snap.frame, issue.name, now, t.trend.min_headlines_each_window,
                             snap.coverage_start)
    report = build_report(summary, trend, t.aggregate, t.trend, a.min_headlines,
                          snap.last_success, now, a.stale_after_minutes)
    return IssueView(issue, summary, trend, report)


def all_issue_views(cfg: AppConfig, snap: Snapshot, window: agg.Window, now: datetime) -> list[IssueView]:
    return [issue_view(cfg, snap, issue, window, now) for issue in cfg.issues.issues]


def format_ist(value: datetime | str | None, with_time: bool = True) -> str:
    """Human-readable IST timestamp, e.g. ``09 Oct 2026, 12:03 IST``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Date not given"
    if isinstance(value, str):
        value = parse_utc_iso(value)
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    local = value.astimezone(IST)
    return local.strftime("%d %b %Y, %H:%M IST" if with_time else "%d %b %Y")


# --------------------------------------------------------------------------- headlines page
HEADLINES_SQL = """
SELECT a.id AS article_id, a.headline, a.source, a.url, a.published_at, a.ingested_at,
       CASE WHEN a.published_at IS NULL OR a.published_at > a.ingested_at
            THEN a.ingested_at ELSE a.published_at END AS event_time,
       c.primary_issue AS issue, c.secondary_issue AS secondary, c.evidence, c.method,
       c.keyword_issue, c.validation_status, c.model_id, c.is_relevant, c.classified_at,
       s.label, s.compound, s.positive, s.negative, s.neutral
FROM articles a
LEFT JOIN classifications c ON c.article_id = a.id
LEFT JOIN sentiment_scores s ON s.article_id = a.id
"""


def load_headlines(cfg: AppConfig) -> pd.DataFrame:
    """All stored headlines including rejected and not-yet-classified ones (for auditing)."""
    with closing(open_db(cfg)) as conn:
        df = pd.read_sql_query(HEADLINES_SQL + " ORDER BY event_time DESC", conn)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, format="ISO8601")
    return df


# --------------------------------------------------------------------------- developer page
def load_developer_tables(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    with closing(open_db(cfg)) as conn:
        q = lambda sql: pd.read_sql_query(sql, conn)  # noqa: E731
        return {
            "runs": q("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 200"),
            "methods": q("""SELECT method, validation_status, COUNT(*) AS headlines,
                                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                                   SUM(from_cache) AS from_cache
                            FROM classifications GROUP BY method, validation_status
                            ORDER BY headlines DESC"""),
            "models": q("""SELECT COALESCE(model_id, '(none — keyword fallback)') AS model_id,
                                  prompt_version, method, COUNT(*) AS headlines,
                                  MIN(classified_at) AS first_used, MAX(classified_at) AS last_used
                           FROM classifications GROUP BY 1, 2, 3 ORDER BY last_used DESC"""),
            "agreement": q("""SELECT method, COUNT(*) AS headlines,
                                     SUM(primary_issue = keyword_issue) AS agree
                              FROM classifications GROUP BY method"""),
            "review": q("""SELECT r.id, r.status, r.reason, r.llm_label, r.keyword_label,
                                  r.corrected_issue, r.reviewer_notes, a.headline, c.evidence,
                                  c.raw_response, r.created_at, r.reviewed_at
                           FROM review_queue r JOIN articles a ON a.id = r.article_id
                           LEFT JOIN classifications c ON c.article_id = r.article_id
                           ORDER BY (r.status = 'pending') DESC, r.id DESC"""),
            "alerts": q("SELECT * FROM alerts ORDER BY id DESC LIMIT 200"),
            "failed": q("""SELECT COUNT(*) AS unclassified FROM articles a
                           LEFT JOIN classifications c ON c.article_id = a.id WHERE c.id IS NULL"""),
        }


def save_review(cfg: AppConfig, review_id: int, status: str, notes: str | None,
                corrected_issue: str | None) -> None:
    """The only write the dashboard performs: recording a human review decision."""
    with closing(open_db(cfg)) as conn:
        with db.transaction(conn):
            db.update_review_item(conn, review_id, status, notes, corrected_issue)


def tail_log(cfg: AppConfig, lines: int = 200) -> str:
    log_file = cfg.settings.logging.file
    if not log_file:
        return ""
    path: Path = cfg.resolve_path(log_file)
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])
