import sqlite3
from datetime import timedelta

import pytest

from voterpulse import database as db
from voterpulse.config import load_config
from voterpulse.ingestion.deduplicator import prepare
from voterpulse.models import (ClassificationMethod, ClassificationOutcome, RawHeadline,
                               SentimentLabel, SentimentScores, ValidationStatus, utc_now)


def _article(text: str = "Farmers protest MSP delay", feed: str = "farmers"):
    return prepare(RawHeadline(text, "https://x.test/a", "Src", None, feed))


def test_config_loads_and_generates_seven_labels(cfg):
    assert len(cfg.issues.issues) == 6
    assert cfg.issues.all_labels[-1] == "Not an election issue"
    assert len(cfg.issues.all_labels) == 7


def test_env_overrides(tmp_path):
    cfg = load_config(env={"VOTERPULSE_DEMO_MODE": "1", "VOTERPULSE_MODEL": "qwen2.5:3b"})
    assert cfg.settings.demo_enabled is True
    assert cfg.model.ollama.model == "qwen2.5:3b"
    assert cfg.db_path == cfg.demo_db_path


def test_schema_created_with_pragmas(conn):
    assert db.schema_version(conn) == len(db.MIGRATIONS)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"articles", "classifications", "sentiment_scores", "review_queue",
            "pipeline_runs", "alerts", "classification_cache"} <= tables


def test_migrate_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    db.init_db(path).close()
    conn = db.init_db(path)
    assert db.schema_version(conn) == len(db.MIGRATIONS)
    conn.close()


def test_demo_and_live_databases_cannot_be_mixed(tmp_path):
    path = tmp_path / "live.db"
    conn = db.init_db(path, "live")
    with pytest.raises(db.DatasetKindError):
        db.insert_article(conn, _article(), is_synthetic=True)
    conn.close()
    with pytest.raises(db.DatasetKindError):
        db.init_db(path, "demo")


def test_duplicate_article_hash_is_ignored(conn):
    assert db.insert_article(conn, _article()) is not None
    assert db.insert_article(conn, _article("FARMERS protest — MSP delay!")) is None
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.save_sentiment(conn, 999, SentimentScores(0, 0, 1, 0, SentimentLabel.NEUTRAL))


def test_sentiment_not_duplicated(conn):
    aid = db.insert_article(conn, _article())
    s = SentimentScores(0.1, 0.2, 0.7, -0.3, SentimentLabel.NEGATIVE)
    assert db.save_sentiment(conn, aid, s) is True
    assert db.save_sentiment(conn, aid, s) is False
    assert conn.execute("SELECT COUNT(*) FROM sentiment_scores").fetchone()[0] == 1


def test_secondary_cannot_equal_primary(conn):
    aid = db.insert_article(conn, _article())
    bad = ClassificationOutcome("Farmers and Agriculture", "Farmers and Agriculture", "MSP",
                                ClassificationMethod.LLM, ValidationStatus.VALID, "Farmers and Agriculture")
    with pytest.raises(sqlite3.IntegrityError):
        db.save_classification(conn, aid, bad, True)


def test_transaction_rolls_back(conn):
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            db.insert_article(conn, _article())
            raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 0


def test_run_lock_prevents_overlap_and_recovers_stale(conn):
    first = db.start_run(conn, "test", stale_after_minutes=30)
    assert first is not None
    assert db.start_run(conn, "test", stale_after_minutes=30) is None
    old = (utc_now() - timedelta(hours=2)).isoformat(timespec="seconds")
    conn.execute("UPDATE pipeline_runs SET started_at = ? WHERE id = ?", (old, first))
    second = db.start_run(conn, "test", stale_after_minutes=30)
    assert second is not None
    status = conn.execute("SELECT status FROM pipeline_runs WHERE id = ?", (first,)).fetchone()[0]
    assert status == "abandoned"


def test_only_one_active_alert_per_issue(conn):
    row = ("Law and Order", "more_negative", "noticeable", 0, -0.2, 0.2, 30, 30, "m", "t", "t", "active")
    sql = """INSERT INTO alerts (issue, direction, severity, previous_sentiment, current_sentiment,
             magnitude, current_count, reference_count, message, triggered_at, last_checked_at, status)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
    conn.execute(sql, row)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, row)
