"""End-to-end ingestion pipeline (one run).

RSS -> clean/hash -> dedupe -> store -> classify (cache/LLM/fallback) -> VADER ->
store -> alerts. Each feed and each article is isolated: one failure is logged and
counted, never fatal to the run. The pipeline always writes to the *live* database;
demo data is generated separately by scripts/generate_demo_data.py.
"""

from __future__ import annotations

import sqlite3
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime

from voterpulse import database as db
from voterpulse.classification.llm_classifier import OllamaClassifier
from voterpulse.classification.service import HeadlineClassifier
from voterpulse.config import AppConfig, RssConfig
from voterpulse.ingestion.deduplicator import deduplicate_batch
from voterpulse.ingestion.rss_fetcher import fetch_all
from voterpulse.logging_utils import fields, get_logger
from voterpulse.models import FeedResult, utc_now
from voterpulse.monitoring.alerts import evaluate_alerts
from voterpulse.sentiment.vader_analyzer import VaderAnalyzer

log = get_logger(__name__)

Fetcher = Callable[[RssConfig], list[FeedResult]]


@dataclass
class RunStats:
    headlines_collected: int = 0
    new_headlines: int = 0
    duplicates: int = 0
    classified: int = 0
    llm_classified: int = 0
    fallback_classified: int = 0
    rejected: int = 0
    failed: int = 0
    feeds_ok: int = 0
    feeds_failed: int = 0
    avg_classification_ms: float | None = None
    duration_seconds: float | None = None
    error_info: str | None = None


@dataclass
class RunReport:
    run_id: int | None
    status: str
    stats: RunStats
    feed_errors: dict[str, str] = field(default_factory=dict)
    llm_note: str | None = None
    backlog: int = 0


def ingest(conn: sqlite3.Connection, feeds: list[FeedResult], stats: RunStats,
           now: datetime | None = None) -> None:
    """Store new, de-duplicated headlines from fetched feeds."""
    raws = [h for f in feeds if f.ok for h in f.headlines]
    stats.headlines_collected = len(raws)
    unique, batch_dupes, unusable = deduplicate_batch(raws)
    stats.duplicates += batch_dupes
    if unusable:
        log.info("headlines_unusable", extra=fields(count=unusable))
    with db.transaction(conn):
        for article in unique:
            if db.insert_article(conn, article, ingested_at=now) is None:
                stats.duplicates += 1
            else:
                stats.new_headlines += 1


def classify_and_score(conn: sqlite3.Connection, cfg: AppConfig, classifier: HeadlineClassifier,
                       vader: VaderAnalyzer, stats: RunStats, limit: int) -> None:
    """Classify pending articles (oldest first) and score the accepted ones."""
    not_relevant = cfg.issues.not_relevant_label
    latencies: list[float] = []
    for row in db.unclassified_articles(conn, limit):
        try:
            outcome = classifier.classify(row["headline"], row["headline_hash"])
            relevant = outcome.primary_issue != not_relevant
            with db.transaction(conn):
                db.save_classification(conn, row["id"], outcome, relevant)
                for reason in outcome.review_reasons:
                    llm_label = outcome.primary_issue if outcome.method.value == "llm" else None
                    db.add_review_item(conn, row["id"], llm_label, outcome.keyword_issue, reason)
                if relevant:
                    db.save_sentiment(conn, row["id"], vader.score(row["headline"]))
            stats.classified += 1
            if outcome.method.value == "llm":
                stats.llm_classified += 1
                if outcome.latency_ms is not None:
                    latencies.append(outcome.latency_ms)
            else:
                stats.fallback_classified += 1
            if not relevant:
                stats.rejected += 1
        except Exception:
            stats.failed += 1
            log.exception("article_failed", extra=fields(article_id=row["id"]))
    if latencies:
        stats.avg_classification_ms = round(sum(latencies) / len(latencies), 1)

    # Safety net: score any accepted article that somehow lacks a score.
    for row in db.relevant_unscored_articles(conn):
        try:
            db.save_sentiment(conn, row["id"], vader.score(row["headline"]))
        except Exception:
            log.exception("sentiment_failed", extra=fields(article_id=row["id"]))


def reclassify_fallbacks(conn: sqlite3.Connection) -> int:
    """Delete keyword-fallback classifications (LLM unavailable, or LLM answer invalid) so
    the next run re-classifies them with the LLM — e.g. after Ollama is started or the
    prompt is improved. Sentiment rows are kept (they do not depend on the issue) but only
    count once the article is re-accepted."""
    with db.transaction(conn):
        ids = [r[0] for r in conn.execute(
            """SELECT article_id FROM classifications
               WHERE method = 'keyword_fallback'
                 AND validation_status IN ('llm_unavailable', 'llm_invalid')""")]
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            conn.execute(f"DELETE FROM review_queue WHERE status = 'pending' AND article_id IN ({marks})", chunk)
            conn.execute(f"DELETE FROM classifications WHERE article_id IN ({marks})", chunk)
    log.info("fallbacks_queued_for_reclassification", extra=fields(count=len(ids)))
    return len(ids)


def run_pipeline(cfg: AppConfig, conn: sqlite3.Connection | None = None, trigger: str = "manual",
                 fetcher: Fetcher | None = None, llm: OllamaClassifier | None | bool = True,
                 now: datetime | None = None) -> RunReport:
    """Execute one full pipeline run and record it in ``pipeline_runs``.

    ``llm=True`` builds an Ollama client from config, ``None``/``False`` forces the keyword
    fallback, or pass a ready client (tests). ``fetcher`` defaults to Google News RSS.
    """
    own_conn = conn is None
    if conn is None:
        conn = db.init_db(cfg.live_db_path, "live", cfg.settings.database.busy_timeout_ms)
    pcfg = cfg.settings.pipeline
    stats = RunStats()
    run_id = db.start_run(conn, trigger, pcfg.stale_run_minutes)
    if run_id is None:
        log.warning("pipeline_skipped", extra=fields(reason="another run is in progress"))
        if own_conn:
            conn.close()
        return RunReport(None, "skipped", stats)

    started = time.perf_counter()
    report = RunReport(run_id, "running", stats)
    log.info("pipeline_started", extra=fields(run_id=run_id, trigger=trigger))
    try:
        feeds = (fetcher or fetch_all)(cfg.rss)
        stats.feeds_ok = sum(f.ok for f in feeds)
        stats.feeds_failed = sum(not f.ok for f in feeds)
        report.feed_errors = {f.feed_key: f.error or "" for f in feeds if not f.ok}
        ingest(conn, feeds, stats, now)

        if llm is True:
            llm_client: OllamaClassifier | None = OllamaClassifier.from_config(cfg)
        elif llm is False or llm is None:
            llm_client = None
        else:
            llm_client = llm
        classifier = HeadlineClassifier(cfg.issues, llm_client, conn, pcfg.llm_circuit_breaker_failures)
        classify_and_score(conn, cfg, classifier, VaderAnalyzer(cfg.thresholds.headline), stats,
                           pcfg.max_classifications_per_run)
        report.llm_note = classifier.llm_disabled_reason
        evaluate_alerts(conn, cfg, now or utc_now())
        report.backlog = db.count_unclassified(conn)

        notes = []
        if report.feed_errors:
            notes.append("feeds failed: " + "; ".join(f"{k}: {v}" for k, v in report.feed_errors.items()))
        if report.llm_note:
            notes.append(f"LLM unavailable, keyword fallback used ({report.llm_note})")
        stats.error_info = " | ".join(notes) or None
        if stats.feeds_ok == 0:
            report.status = "failed"
        elif stats.feeds_failed or stats.failed:
            report.status = "partial"
        else:
            report.status = "success"
    except Exception as exc:
        report.status = "failed"
        stats.error_info = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        log.exception("pipeline_failed", extra=fields(run_id=run_id))
    finally:
        stats.duration_seconds = round(time.perf_counter() - started, 2)
        try:
            db.finish_run(conn, run_id, report.status, asdict(stats))
        except Exception:
            log.exception("run_record_failed", extra=fields(run_id=run_id))
        log.info("pipeline_finished", extra=fields(run_id=run_id, status=report.status,
                                                   **{k: v for k, v in asdict(stats).items()
                                                      if k != "error_info"}))
        if own_conn:
            conn.close()
    return report
