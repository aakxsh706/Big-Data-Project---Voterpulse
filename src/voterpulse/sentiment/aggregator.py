"""Issue-level sentiment aggregation over consistent time windows.

Time basis: each headline's *event time* is its publication time, or its ingestion
time when the feed gave no date (or a publication time later than ingestion, which
indicates a bad clock). All windows are half-open ``[start, end)`` in UTC.

Only primary classifications of accepted (relevant) headlines are aggregated, so a
headline is counted exactly once and secondary labels never inflate counts.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from voterpulse.config import IST
from voterpulse.models import parse_utc_iso, to_utc_iso

PERIODS: dict[str, timedelta] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
PERIOD_LABELS = {"24h": "Last 24 hours", "7d": "Last 7 days", "30d": "Last 30 days"}

RELEVANT_SQL = """
SELECT a.id AS article_id, a.headline, a.source, a.url, a.published_at, a.ingested_at,
       CASE WHEN a.published_at IS NULL OR a.published_at > a.ingested_at
            THEN a.ingested_at ELSE a.published_at END AS event_time,
       a.is_synthetic,
       c.primary_issue AS issue, c.secondary_issue AS secondary, c.evidence, c.method,
       c.validation_status, c.model_id, c.keyword_issue,
       s.compound, s.positive, s.negative, s.neutral, s.label
FROM articles a
JOIN classifications c ON c.article_id = a.id AND c.is_relevant = 1
JOIN sentiment_scores s ON s.article_id = a.id
"""

FRAME_COLUMNS = ["article_id", "headline", "source", "url", "published_at", "ingested_at",
                 "event_time", "is_synthetic", "issue", "secondary", "evidence", "method",
                 "validation_status", "model_id", "keyword_issue", "compound", "positive",
                 "negative", "neutral", "label"]


def load_relevant_frame(conn: sqlite3.Connection, since: datetime | None = None) -> pd.DataFrame:
    """All accepted, scored headlines (optionally with event time >= ``since``)."""
    sql, params = RELEVANT_SQL, []
    if since is not None:
        sql = f"SELECT * FROM ({RELEVANT_SQL}) WHERE event_time >= ?"
        params.append(to_utc_iso(since))
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        df = pd.DataFrame(columns=FRAME_COLUMNS)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, format="ISO8601")
    df["compound"] = pd.to_numeric(df["compound"], errors="coerce").astype(float)
    return df


def coverage_start(conn: sqlite3.Connection) -> datetime | None:
    """When collection began (earliest ingestion) — earlier windows are incomplete."""
    row = conn.execute("SELECT MIN(ingested_at) FROM articles").fetchone()
    return parse_utc_iso(row[0]) if row and row[0] else None


# --------------------------------------------------------------------------- windows
@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    key: str
    label: str
    is_partial: bool = False

    @property
    def days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400


def rolling_window(period: str, now: datetime, data_start: datetime | None = None) -> Window:
    """Window of length ``period`` ending at ``now``.

    Marked partial when collection started after the window began.
    """
    if period not in PERIODS:
        raise ValueError(f"unknown period {period!r}; use one of {list(PERIODS)}")
    start = now - PERIODS[period]
    partial = data_start is not None and data_start > start
    return Window(start, now, period, PERIOD_LABELS[period], partial)


def custom_window(start_day: date, end_day: date, now: datetime,
                  data_start: datetime | None = None) -> Window:
    """Inclusive IST calendar dates -> UTC window. Partial if it extends past ``now``."""
    if end_day < start_day:
        raise ValueError("end date is before start date")
    start = datetime.combine(start_day, time.min, IST).astimezone(timezone.utc)
    end = datetime.combine(end_day + timedelta(days=1), time.min, IST).astimezone(timezone.utc)
    partial = end > now or (data_start is not None and data_start > start)
    label = f"{start_day:%d %b %Y} – {end_day:%d %b %Y}"
    return Window(start, min(end, now) if end > now else end, "custom", label, partial)


def ist_midnight(now: datetime) -> datetime:
    """Start of the current IST calendar day, in UTC."""
    local = now.astimezone(IST)
    return datetime.combine(local.date(), time.min, IST).astimezone(timezone.utc)


def completed_week_windows(now: datetime, data_start: datetime | None = None) -> tuple[Window, Window]:
    """The most recent completed 7-day window and the 7 days before it.

    Both end on an IST midnight, so today's partial day is excluded from the trend.
    """
    end = ist_midnight(now)
    current = Window(end - timedelta(days=7), end, "week", "Previous 7 complete days",
                     data_start is not None and data_start > end - timedelta(days=7))
    previous = Window(end - timedelta(days=14), end - timedelta(days=7), "prev_week",
                      "The 7 days before that",
                      data_start is not None and data_start > end - timedelta(days=14))
    return current, previous


def in_window(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    if df.empty:
        return df
    start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
    return df[(df["event_time"] >= start) & (df["event_time"] < end)]


# --------------------------------------------------------------------------- summaries
@dataclass
class IssueSummary:
    issue: str
    window: Window
    count: int
    mean_compound: float | None
    positive_count: int
    negative_count: int
    neutral_count: int
    sufficient: bool
    latest_event_time: datetime | None
    sources: list[tuple[str, int]] = field(default_factory=list)

    def share(self, n: int) -> float:
        return n / self.count if self.count else 0.0

    @property
    def positive_share(self) -> float:
        return self.share(self.positive_count)

    @property
    def negative_share(self) -> float:
        return self.share(self.negative_count)

    @property
    def neutral_share(self) -> float:
        return self.share(self.neutral_count)


def summarize(df: pd.DataFrame, issue: str, window: Window, min_headlines: int) -> IssueSummary:
    sub = in_window(df, window)
    sub = sub[sub["issue"] == issue] if not sub.empty else sub
    count = int(len(sub))
    labels = Counter(sub["label"]) if count else Counter()
    sources = Counter(s if isinstance(s, str) and s else "Unknown source" for s in sub["source"]) if count else Counter()
    return IssueSummary(
        issue=issue,
        window=window,
        count=count,
        mean_compound=float(sub["compound"].mean()) if count else None,
        positive_count=labels.get("Positive", 0),
        negative_count=labels.get("Negative", 0),
        neutral_count=labels.get("Neutral", 0),
        sufficient=count >= min_headlines,
        latest_event_time=sub["event_time"].max().to_pydatetime() if count else None,
        sources=sorted(sources.items(), key=lambda kv: (-kv[1], kv[0])),
    )


@dataclass
class TrendResult:
    issue: str
    current: Window
    previous: Window
    current_count: int
    previous_count: int
    current_mean: float | None
    previous_mean: float | None
    sufficient: bool

    @property
    def delta(self) -> float | None:
        if not self.sufficient or self.current_mean is None or self.previous_mean is None:
            return None
        return self.current_mean - self.previous_mean


def weekly_trend(df: pd.DataFrame, issue: str, now: datetime, min_each: int,
                 data_start: datetime | None = None) -> TrendResult:
    """Compare the last completed 7-day window with the 7 days before it."""
    current_w, previous_w = completed_week_windows(now, data_start)
    cur = summarize(df, issue, current_w, min_each)
    prev = summarize(df, issue, previous_w, min_each)
    return TrendResult(
        issue=issue, current=current_w, previous=previous_w,
        current_count=cur.count, previous_count=prev.count,
        current_mean=cur.mean_compound, previous_mean=prev.mean_compound,
        sufficient=cur.sufficient and prev.sufficient and not previous_w.is_partial,
    )


def daily_series(df: pd.DataFrame, issue: str, window: Window) -> pd.DataFrame:
    """Per-IST-day headline counts and mean compound, for trend charts."""
    sub = in_window(df, window)
    sub = sub[sub["issue"] == issue] if not sub.empty else sub
    columns = ["day", "headlines", "mean_compound", "Positive", "Neutral", "Negative"]
    if sub.empty:
        return pd.DataFrame(columns=columns)
    sub = sub.assign(day=sub["event_time"].dt.tz_convert(IST).dt.date)
    grouped = sub.groupby("day").agg(headlines=("article_id", "count"),
                                     mean_compound=("compound", "mean"))
    labels = sub.pivot_table(index="day", columns="label", values="article_id",
                             aggfunc="count", fill_value=0)
    out = grouped.join(labels).reset_index()
    for col in ("Positive", "Neutral", "Negative"):
        if col not in out:
            out[col] = 0
    return out[columns]


def mean_in(df: pd.DataFrame, issue: str, start: datetime, end: datetime) -> tuple[float | None, int]:
    sub = in_window(df, Window(start, end, "adhoc", "adhoc"))
    sub = sub[sub["issue"] == issue] if not sub.empty else sub
    return (float(sub["compound"].mean()) if len(sub) else None), int(len(sub))
