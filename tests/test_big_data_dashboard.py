"""Offline checks for batch isolation, bounded Mongo reads, and dashboard states."""

from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from voterpulse.config import load_config
from voterpulse.dashboard import big_data

NOW = datetime(2026, 10, 11, 20, 0, tzinfo=timezone.utc)  # 12 October in IST


@pytest.fixture
def cfg():
    config = load_config(env={})
    config.settings.big_data.enabled = True
    config.settings.demo_enabled = False
    return config


def _summary(day="2026-10-12", issue="Healthcare", count=3):
    return {
        "day": day, "issue": issue, "article_count": count, "positive_count": 1,
        "neutral_count": count - 1, "negative_count": 0, "mean_compound": 0.15, "source_count": 2,
    }


def _run():
    return {
        "_id": "completed-run", "run_id": "completed-run", "status": "success",
        "dataset_kind": "live", "completed_at": NOW, "input_count": 8, "eligible_count": 3,
        "summary_count": 1, "partitions": 2, "spark_version": "3.5.8", "spark_master": "local[2]",
        "duration_seconds": 12.4,
    }


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.projection = None
        self.sort_order = None
        self.read_limit = None

    def find_one(self, query, projection, sort):
        self.query, self.projection, self.sort_order = query, projection, sort
        return self.rows[0] if self.rows else None

    def find(self, query, projection):
        self.query, self.projection = query, projection
        return self

    def sort(self, order):
        self.sort_order = order
        return self

    def limit(self, count):
        self.read_limit = count
        return self

    def __iter__(self):
        assert self.read_limit is not None, "The MongoDB query must be bounded before iteration"
        return iter(self.rows[:self.read_limit])


def _mongo(monkeypatch, run, rows):
    collections = {
        "spark_runs": FakeCollection([run] if run else []),
        "daily_summaries": FakeCollection(rows),
    }
    requested_kinds = []

    @contextmanager
    def open_mongo(config, kind):
        requested_kinds.append(kind)
        yield collections

    monkeypatch.setattr(big_data, "_open_mongo", open_mongo)
    return collections, requested_kinds


def test_disabled_profile_does_not_open_mongo(cfg, monkeypatch):
    cfg.settings.big_data.enabled = False

    def unexpected_connection(*args):
        pytest.fail("Disabled profiles must not import the driver or open MongoDB")

    monkeypatch.setattr(big_data, "_open_mongo", unexpected_connection)
    snapshot = big_data.load_big_data_snapshot(cfg, NOW)
    assert snapshot.status == "disabled"
    assert snapshot.summaries.empty


@pytest.mark.parametrize("demo,kind", [(False, "live"), (True, "demo")])
def test_reads_latest_successful_run_in_matching_database(cfg, monkeypatch, demo, kind):
    cfg.settings.demo_enabled = demo
    collections, kinds = _mongo(monkeypatch, _run(), [_summary()])
    snapshot = big_data.load_big_data_snapshot(cfg, NOW)
    assert snapshot.status == "ready"
    assert kinds == [kind]
    assert snapshot.database == (
        cfg.settings.big_data.mongo_demo_database if demo else cfg.settings.big_data.mongo_database
    )
    assert collections["spark_runs"].query == {"status": "success", "dataset_kind": kind}
    assert collections["spark_runs"].sort_order == [("completed_at", -1)]
    query = collections["daily_summaries"].query
    assert query["run_id"] == "completed-run"
    assert query["dataset_kind"] == kind
    assert query["day"] == {"$gte": "2026-07-15", "$lte": "2026-10-12"}
    assert "error" not in collections["spark_runs"].projection
    assert list(snapshot.summaries.columns) == big_data.SUMMARY_COLUMNS
    assert snapshot.summaries.iloc[0]["article_count"] == 3


def test_summary_query_is_bounded_and_signals_truncation(cfg, monkeypatch):
    collections, _ = _mongo(monkeypatch, _run(), [_summary() for _ in range(7)])
    snapshot = big_data.load_big_data_snapshot(cfg, NOW, max_rows=2)
    assert collections["daily_summaries"].read_limit == 3
    assert len(snapshot.summaries) == 2
    assert snapshot.truncated
    assert collections["daily_summaries"].sort_order == [("day", -1), ("issue", 1)]


def test_empty_run_history_does_not_read_summary_collection(cfg, monkeypatch):
    collections, _ = _mongo(monkeypatch, None, [_summary()])
    snapshot = big_data.load_big_data_snapshot(cfg, NOW)
    assert snapshot.status == "empty"
    assert collections["daily_summaries"].query is None


@pytest.mark.parametrize("error", [ModuleNotFoundError, ConnectionError])
def test_driver_and_connection_failures_do_not_disclose_uri(cfg, monkeypatch, error):
    def unavailable(*args):
        raise error("mongodb://user:private-password@unreachable.example/db")

    monkeypatch.setattr(big_data, "_open_mongo", unavailable)
    snapshot = big_data.load_big_data_snapshot(cfg, NOW)
    assert snapshot.status == "unavailable"
    assert snapshot.error_type == error.__name__
    assert "private-password" not in repr(snapshot)


def _render_panel():
    from voterpulse.config import get_config
    from voterpulse.dashboard.big_data import render_big_data

    render_big_data(get_config())


def _app(cfg, monkeypatch, snapshot):
    from voterpulse import config

    monkeypatch.setattr(config, "get_config", lambda: cfg)
    monkeypatch.setattr(big_data, "load_big_data_snapshot", lambda *args, **kwargs: snapshot)
    return AppTest.from_function(_render_panel, default_timeout=15).run()


@pytest.mark.parametrize("status", ["disabled", "empty", "unavailable"])
def test_empty_and_error_panels_explain_next_step(cfg, monkeypatch, status):
    snapshot = big_data.BigDataSnapshot(
        status=status, dataset_kind="demo", database="voterpulse_demo", error_type="ConnectionError",
    )
    app = _app(cfg, monkeypatch, snapshot)
    assert not app.exception
    assert any("python scripts/run_big_data.py --demo" in block.value for block in app.code)
    assert app.warning if status == "unavailable" else app.info


def test_success_panel_shows_real_batch_metrics_and_summaries(cfg, monkeypatch):
    snapshot = big_data.BigDataSnapshot(
        status="ready", dataset_kind="live", database="voterpulse", run=_run(),
        summaries=pd.DataFrame([_summary()]), window_start="2026-07-15", window_end="2026-10-12",
    )
    app = _app(cfg, monkeypatch, snapshot)
    assert not app.exception, [error.value for error in app.exception]
    assert {metric.label: metric.value for metric in app.metric} == {
        "Articles read by Spark": "8", "Eligible headlines": "3", "Spark partitions": "2",
        "Batch duration": "12.4s",
    }
    captions = " ".join(item.value for item in app.caption)
    assert "Spark 3.5.8" in captions
    assert "12 Oct 2026, 01:30 IST" in captions
    assert "local Spark on one computer" in captions
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.iloc[0]["article_count"] == 3
    assert any("local SQLite" in item.value for item in app.markdown)
