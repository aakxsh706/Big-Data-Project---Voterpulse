"""Mood-shift alerts.

Method (documented in docs/methodology.md):

* Current window: the rolling last 24 hours (``alerts.current_window_hours``).
* Reference window: the 7 days immediately before the current window — no overlap.
* delta = mean compound(current) - mean compound(reference).
* Trigger when both windows meet the minimum sample size and |delta| >= trigger_delta.
  Severity is "marked" when |delta| >= large_delta, otherwise "noticeable".
* Duplicate prevention: at most one *active* alert per issue (enforced by a unique
  partial index). While the shift continues in the same direction the existing alert
  is updated, not re-created.
* Reset (hysteresis): the alert resolves when |delta| falls below reset_delta, when the
  direction flips, or when either window drops below the minimum sample size. A
  flip beyond the trigger immediately opens a new alert in the new direction.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from voterpulse import database as db
from voterpulse.config import AppConfig
from voterpulse.logging_utils import fields, get_logger
from voterpulse.models import to_utc_iso
from voterpulse.sentiment.aggregator import load_relevant_frame, mean_in

log = get_logger(__name__)


@dataclass
class AlertCheck:
    issue: str
    current_mean: float | None
    reference_mean: float | None
    current_count: int
    reference_count: int
    sufficient: bool
    delta: float | None
    action: str  # none | created | updated | resolved | resolved_and_created | insufficient


def severity_for(magnitude: float, large_delta: float) -> str:
    return "marked" if magnitude >= large_delta else "noticeable"


def alert_message(issue: str, direction: str, severity: str) -> str:
    topic = issue.lower()
    if direction == "more_negative":
        lead = "Noticeable" if severity == "noticeable" else "Marked"
        return (f"{lead} increase in negative coverage about {topic} in the last 24 hours, "
                f"compared with the previous week.")
    adverb = "noticeably" if severity == "noticeable" else "markedly"
    return (f"{issue} coverage has become {adverb} more positive in the last 24 hours, "
            f"compared with the previous week.")


def evaluate_alerts(conn: sqlite3.Connection, cfg: AppConfig, now: datetime,
                    df: pd.DataFrame | None = None) -> list[AlertCheck]:
    """Check every issue and create/update/resolve alerts. Writes in one transaction."""
    a = cfg.settings.alerts
    current_start = now - timedelta(hours=a.current_window_hours)
    reference_start = current_start - timedelta(days=a.reference_window_days)
    if df is None:
        df = load_relevant_frame(conn, since=reference_start)
    checks: list[AlertCheck] = []
    stamp = to_utc_iso(now)

    with db.transaction(conn):
        for issue in cfg.issues.issue_names:
            cur_mean, cur_n = mean_in(df, issue, current_start, now)
            ref_mean, ref_n = mean_in(df, issue, reference_start, current_start)
            sufficient = (cur_n >= a.min_current_headlines and ref_n >= a.min_reference_headlines
                          and cur_mean is not None and ref_mean is not None)
            delta = (cur_mean - ref_mean) if sufficient else None
            active = db.active_alert(conn, issue)
            action = "none"

            if not sufficient:
                action = "insufficient"
                if active is not None:
                    _resolve(conn, active["id"], stamp)
                    action = "resolved"
            else:
                assert delta is not None
                magnitude = abs(delta)
                direction = "more_positive" if delta > 0 else "more_negative"
                if active is not None:
                    if active["direction"] == direction and magnitude >= a.reset_delta:
                        severity = severity_for(magnitude, a.large_delta)
                        conn.execute(
                            """UPDATE alerts SET current_sentiment = ?, previous_sentiment = ?,
                                      magnitude = ?, severity = ?, current_count = ?,
                                      reference_count = ?, message = ?, last_checked_at = ?
                               WHERE id = ?""",
                            (cur_mean, ref_mean, magnitude, severity, cur_n, ref_n,
                             alert_message(issue, direction, severity), stamp, active["id"]),
                        )
                        action = "updated"
                    else:
                        _resolve(conn, active["id"], stamp)
                        action = "resolved"
                        active = None
                if active is None and magnitude >= a.trigger_delta:
                    severity = severity_for(magnitude, a.large_delta)
                    conn.execute(
                        """INSERT INTO alerts (issue, direction, severity, previous_sentiment,
                                  current_sentiment, magnitude, current_count, reference_count,
                                  message, triggered_at, last_checked_at, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
                        (issue, direction, severity, ref_mean, cur_mean, magnitude, cur_n, ref_n,
                         alert_message(issue, direction, severity), stamp, stamp),
                    )
                    action = "resolved_and_created" if action == "resolved" else "created"
            if action in {"created", "resolved", "resolved_and_created"}:
                log.info("alert_" + action, extra=fields(issue=issue, delta=delta,
                                                         current_n=cur_n, reference_n=ref_n))
            checks.append(AlertCheck(issue, cur_mean, ref_mean, cur_n, ref_n, sufficient, delta, action))
    return checks


def _resolve(conn: sqlite3.Connection, alert_id: int, stamp: str | None) -> None:
    conn.execute(
        "UPDATE alerts SET status = 'resolved', resolved_at = ?, last_checked_at = ? WHERE id = ?",
        (stamp, stamp, alert_id),
    )
