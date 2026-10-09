"""Independent 15-minute scheduler (runs in its own process, never inside Streamlit).

``max_instances=1`` + ``coalesce=True`` mean a slow run is never overlapped by the
next tick, and missed ticks collapse into one. The database run lock in
:func:`voterpulse.database.start_run` additionally prevents overlap with a manual
``run_pipeline.py`` started from another terminal.
"""

from __future__ import annotations

from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from voterpulse.config import AppConfig
from voterpulse.logging_utils import fields, get_logger
from voterpulse.monitoring.pipeline import run_pipeline

log = get_logger(__name__)


def scheduled_job(cfg: AppConfig) -> None:
    try:
        report = run_pipeline(cfg, trigger="scheduler")
        log.info("scheduled_run_done", extra=fields(status=report.status, run_id=report.run_id))
    except Exception:  # keep the scheduler alive no matter what
        log.exception("scheduled_run_crashed")


def build_scheduler(cfg: AppConfig, run_immediately: bool = True) -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=timezone.utc)
    # Note: passing next_run_time=None to APScheduler 3 would add the job *paused*.
    extra = {"next_run_time": datetime.now(timezone.utc)} if run_immediately else {}
    scheduler.add_job(
        scheduled_job,
        IntervalTrigger(minutes=cfg.settings.pipeline.interval_minutes),
        args=[cfg],
        id="voterpulse_pipeline",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
        replace_existing=True,
        **extra,
    )
    return scheduler


def run_forever(cfg: AppConfig, run_immediately: bool = True) -> None:
    scheduler = build_scheduler(cfg, run_immediately)
    log.info("scheduler_started", extra=fields(interval_minutes=cfg.settings.pipeline.interval_minutes))
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler_stopped")
