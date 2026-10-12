"""Read-only dashboard view of MongoDB documents and completed Spark batches.

The optional database driver is imported only when the profile is enabled. The
dashboard never launches Spark or loads the article corpus into pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from voterpulse.config import IST, AppConfig

SUMMARY_COLUMNS = [
    "day", "issue", "article_count", "positive_count", "neutral_count", "negative_count",
    "mean_compound", "source_count",
]
RUN_COLUMNS = [
    "run_id", "dataset_kind", "status", "completed_at", "input_count", "eligible_count",
    "summary_count", "partitions", "spark_version", "spark_master", "duration_seconds",
]


@dataclass
class BigDataSnapshot:
    status: str
    dataset_kind: str
    database: str
    run: dict[str, Any] | None = None
    summaries: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SUMMARY_COLUMNS))
    window_start: str = ""
    window_end: str = ""
    truncated: bool = False
    error_type: str | None = None


def _open_mongo(cfg: AppConfig, kind: str):
    # Keep even the storage module out of the default dashboard startup path.
    from voterpulse.big_data.mongo_store import open_mongo

    return open_mongo(cfg, kind)


def load_big_data_snapshot(
    cfg: AppConfig, now: datetime | None = None, *, days: int = 90, max_rows: int = 1000,
) -> BigDataSnapshot:
    """Fetch one successful batch and a bounded set of its daily aggregates.

    Limit the database query itself, not just the rendered frame. A failed or
    still-running batch must never expose partially written summary documents.
    Connection errors are represented by type only: driver exception messages
    can contain connection strings and credentials.
    """
    settings = cfg.settings.big_data
    kind = "demo" if cfg.settings.demo_enabled else "live"
    database = settings.mongo_demo_database if kind == "demo" else settings.mongo_database
    snapshot = BigDataSnapshot(status="disabled", dataset_kind=kind, database=database)
    if not settings.enabled:
        return snapshot

    days = max(1, min(int(days), 90))
    max_rows = max(1, min(int(max_rows), 1000))
    today = (now or datetime.now(timezone.utc)).astimezone(IST).date()
    snapshot.window_start = (today - timedelta(days=days - 1)).isoformat()
    snapshot.window_end = today.isoformat()
    try:
        with _open_mongo(cfg, kind) as database_handle:
            run = database_handle["spark_runs"].find_one(
                {"status": "success", "dataset_kind": kind},
                {column: 1 for column in RUN_COLUMNS},
                sort=[("completed_at", -1)],
            )
            if not run:
                snapshot.status = "empty"
                return snapshot
            run_id = run.get("run_id") or run.get("_id")
            if not run_id:
                raise ValueError("The completed batch has no run identifier")
            cursor = database_handle["daily_summaries"].find(
                {
                    "run_id": run_id,
                    "dataset_kind": kind,
                    "day": {"$gte": snapshot.window_start, "$lte": snapshot.window_end},
                },
                {"_id": 0, **{column: 1 for column in SUMMARY_COLUMNS}},
            ).sort([("day", -1), ("issue", 1)]).limit(max_rows + 1)
            rows = list(cursor)
        snapshot.status = "ready"
        snapshot.run = {key: run[key] for key in RUN_COLUMNS if key in run}
        snapshot.truncated = len(rows) > max_rows
        snapshot.summaries = pd.DataFrame(rows[:max_rows], columns=SUMMARY_COLUMNS)
    except Exception as exc:
        snapshot.status = "unavailable"
        snapshot.error_type = type(exc).__name__
    return snapshot


def _batch_command(kind: str) -> str:
    return "python scripts/run_big_data.py" + (" --demo" if kind == "demo" else "")


def render_big_data(cfg: AppConfig, now: datetime | None = None) -> None:
    """Render batch results without starting any background processing."""
    import streamlit as st

    st.subheader("MongoDB + Apache Spark")
    st.write(
        "MongoDB stores article documents. A separate Spark batch job reads those documents "
        "and computes daily sentiment summaries in parallel. This tab reads the latest "
        "successful Spark batch from MongoDB; the Overview and issue pages use local SQLite data."
    )
    snapshot = load_big_data_snapshot(cfg, now)
    st.caption(f"Dataset: {snapshot.dataset_kind} · MongoDB database: {snapshot.database}")
    command = _batch_command(snapshot.dataset_kind)
    if snapshot.status == "disabled":
        st.info(
            "The big-data profile is disabled. Install the optional dependencies and start MongoDB "
            "using the README setup, then enable VOTERPULSE_BIG_DATA_ENABLED=1 in .env and restart "
            "the dashboard. Run the batch from the project folder:"
        )
        st.code(
            "python -m pip install -r requirements-big-data.txt\ndocker compose up -d mongodb\n" + command,
            language="bash",
        )
        return
    if snapshot.status == "unavailable":
        st.warning(
            f"MongoDB batch results are unavailable ({snapshot.error_type}). Check the optional "
            "big-data dependencies, MongoDB service, and connection settings in .env. "
            "After fixing the connection, rerun the batch and refresh this page."
        )
        st.code(command, language="bash")
        return
    if snapshot.status == "empty":
        st.info("No successful Spark batch is available for this dataset. Sync articles and run Spark:")
        st.code(command, language="bash")
        return

    run = snapshot.run or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles read by Spark", run.get("input_count", "—"))
    c2.metric("Eligible headlines", run.get("eligible_count", "—"))
    c3.metric("Spark partitions", run.get("partitions", "—"))
    duration = run.get("duration_seconds")
    c4.metric("Batch duration", f"{duration:.1f}s" if isinstance(duration, (int, float)) else "—")
    completed = run.get("completed_at")
    if isinstance(completed, datetime):
        # PyMongo returns naive UTC unless tz_aware was configured on the client.
        completed = (completed if completed.tzinfo else completed.replace(tzinfo=timezone.utc))
        completed_label = completed.astimezone(IST).strftime("%d %b %Y, %H:%M IST")
    else:
        completed_label = "unrecorded"
    st.caption(
        f"Spark {run.get('spark_version', 'unknown')} · completed {completed_label} · "
        f"{run.get('summary_count', '—')} daily issue summaries in this batch"
    )
    master = str(run.get("spark_master", ""))
    if master.startswith("local"):
        st.caption(
            "Execution: local Spark on one computer. Partitions can run in parallel across local "
            "CPU cores; this run does not demonstrate a multi-machine cluster."
        )
    elif master:
        st.caption("Execution: Spark configured with an external cluster manager.")

    st.markdown("**Daily issue summaries**")
    st.caption(f"IST dates: {snapshot.window_start} through {snapshot.window_end} · latest successful batch only")
    if snapshot.truncated:
        st.warning("Showing the latest 1,000 summary rows. The full batch remains in MongoDB.")
    if snapshot.summaries.empty:
        st.info("This successful batch has no eligible headline summaries in the displayed date window.")
    else:
        counts = snapshot.summaries.pivot_table(
            index="day", columns="issue", values="article_count", aggfunc="sum", fill_value=0,
        ).sort_index()
        st.bar_chart(counts, x_label="Day (IST)", y_label="Headlines", stack=True)
        st.dataframe(snapshot.summaries, hide_index=True, width="stretch")
        st.caption(
            "Spark scores classified headlines with VADER and excludes articles marked out of region. "
            "Mean compound is the average headline sentiment score, not voter opinion."
        )
    with st.expander("Refresh the batch"):
        st.write("Run this command in the project folder, then refresh the page. The dashboard only reads results.")
        st.code(command, language="bash")
