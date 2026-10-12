"""Deterministic Spark calculations; real local Spark tests need the optional extra and Java."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from voterpulse.big_data.spark_job import (
    SUMMARY_COLUMNS, _package_workers, article_schema, connector_uri, redact_error,
    score_headline, summarize_articles,
)
from voterpulse.sentiment.vader_analyzer import VaderAnalyzer


@pytest.mark.parametrize("headline", ["", None, "Wonderful victory brings hope and prosperity",
                                     "Terrible violent disaster leaves people suffering", "Council met today"])
def test_worker_sentiment_matches_application(headline, cfg):
    expected = VaderAnalyzer(cfg.thresholds.headline).score(headline or "")
    compound, label = score_headline(headline, cfg.thresholds.headline.positive_min,
                                    cfg.thresholds.headline.negative_max)
    assert compound == expected.compound
    assert label == expected.label.value


def test_worker_thresholds_include_boundaries():
    positive, _ = score_headline("Wonderful success")
    negative, _ = score_headline("Terrible disaster")
    assert score_headline("Wonderful success", positive_min=positive)[1] == "Positive"
    assert score_headline("Terrible disaster", negative_max=negative)[1] == "Negative"
    assert score_headline("Wonderful success", positive_min=1.1)[1] == "Neutral"


def test_connector_cannot_override_demo_database_with_uri_path():
    uri = connector_uri("mongodb://reader:p%40ss@host:27017/auth_db?retryWrites=true")
    parsed = urlsplit(uri)
    assert parsed.path == "/"
    assert parsed.netloc == "reader:p%40ss@host:27017"
    assert parse_qs(parsed.query) == {"retryWrites": ["true"], "authSource": ["auth_db"]}
    assert parse_qs(urlsplit(connector_uri(
        "mongodb+srv://host/live?authSource=admin&tls=true"
    )).query)["authSource"] == ["admin"]


def test_connector_uri_cannot_disable_acknowledgement_or_timeout():
    uri = connector_uri(
        "mongodb://host/live?w=0&WTimeoutMS=0&writeConcern.w=0&writeConcern.wTimeoutMS=0"
        "&readPreference=secondaryPreferred&readConcernLevel=majority&tls=true"
    )
    assert parse_qs(urlsplit(uri).query) == {
        "authSource": ["live"], "readPreference": ["secondaryPreferred"],
        "readConcernLevel": ["majority"], "tls": ["true"],
    }


def test_exception_redacts_credentials_and_arbitrary_mongo_uris():
    uri = "mongodb://user:p%40ss@localhost/live"
    error = RuntimeError(f"Failed {uri}; password p@ss, user user; mongodb+srv://other:secret@host/db")
    safe = redact_error(error, uri)
    assert "p@ss" not in safe
    assert "p%40ss" not in safe
    assert "secret" not in safe
    assert "mongodb://" not in safe
    assert "mongodb+srv://" not in safe


@pytest.mark.parametrize("write_fails", [False, True])
def test_publish_only_after_all_summary_writes_succeed(monkeypatch, cfg, write_fails):
    pytest.importorskip("pyspark")
    from pyspark.sql import functions as F
    from voterpulse.big_data import mongo_store, spark_job

    runs = {"old": {"_id": "old", "status": "success"}}
    timeline = []

    class RunCollection:
        def insert_one(self, document):
            runs[document["_id"]] = deepcopy(document)

        def replace_one(self, query, document):
            timeline.append(document["status"])
            runs[query["_id"]] = deepcopy(document)

    @contextmanager
    def mongo(*args, **kwargs):
        yield SimpleNamespace(spark_runs=RunCollection())

    class Frame:
        rdd = SimpleNamespace(getNumPartitions=lambda: 2)

        def repartition(self, *_):
            return self

        def persist(self, *_):
            return self

        def count(self):
            return 1

        def withColumn(self, *_):
            return self

        def unpersist(self):
            pass

    def write(*args):
        assert all(row["status"] != "success" for key, row in runs.items() if key != "old")
        timeline.append("write")
        if write_fails:
            raise RuntimeError("simulated partial write failure")

    session = SimpleNamespace(sparkContext=SimpleNamespace(addPyFile=lambda *_: None),
                              version="3.5.7", stop=lambda: None)
    monkeypatch.setattr(mongo_store, "open_mongo", mongo)
    monkeypatch.setattr(spark_job, "create_spark_session", lambda *_: session)
    monkeypatch.setattr(spark_job, "_read_articles", lambda *_: Frame())
    monkeypatch.setattr(spark_job, "eligible_articles", lambda *_: Frame())
    monkeypatch.setattr(spark_job, "summarize_articles", lambda *_, **__: Frame())
    monkeypatch.setattr(spark_job, "_write_summaries", write)
    for function in ("lit", "sha2", "to_json", "struct"):
        monkeypatch.setattr(F, function, lambda *_: None)
    if write_fails:
        with pytest.raises(spark_job.SparkJobError, match="simulated partial write failure"):
            spark_job.run_spark_job(cfg)
        assert timeline == ["write", "failed"]
        assert len([row for row in runs.values() if row["status"] == "success"]) == 1
    else:
        report = spark_job.run_spark_job(cfg)
        assert report["status"] == "success"
        assert timeline == ["write", "success"]
    assert runs["old"] == {"_id": "old", "status": "success"}


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    pytest.importorskip("pyspark", reason="install .[big-data] to exercise distributed calculations")
    java = Path(os.environ["JAVA_HOME"]) / "bin" / "java" if os.environ.get("JAVA_HOME") else shutil.which("java")
    if not java:
        pytest.skip("Java 17 is required for Spark")
    probe = subprocess.run([str(java), "-version"], capture_output=True, timeout=10)
    if probe.returncode:
        pytest.skip("Java 17 is required for Spark")
    from pyspark.sql import SparkSession

    session = (SparkSession.builder.master("local[2]").appName("voterpulse-spark-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.driver.bindAddress", "127.0.0.1")
               .config("spark.driver.host", "127.0.0.1")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.pyspark.python", sys.executable)
               .getOrCreate())
    session.sparkContext.pythonExec = sys.executable
    session.sparkContext.setLogLevel("ERROR")
    session.sparkContext.addPyFile(str(_package_workers(str(tmp_path_factory.mktemp("spark-workers")))))
    yield session
    session.stop()


def _article(key: str, headline: str, when: datetime, **overrides):
    document = {
        "_id": key, "headline": headline, "normalized_headline": headline, "source": "Source A",
        "url": "https://example.test", "issue": "Employment", "region_status": "in_region",
        "dataset_kind": "live", "event_time": when, "published_at": when, "ingested_at": when,
        "is_relevant": True, "is_synthetic": False,
    }
    document.update(overrides)
    return document


def test_parallel_summary_filters_and_ist_boundary(spark):
    before = datetime(2026, 10, 8, 18, 29, tzinfo=timezone.utc)
    after = datetime(2026, 10, 8, 18, 30, tzinfo=timezone.utc)
    positive, negative = "Wonderful success and prosperity", "Terrible disaster and violent failure"
    rows = [
        _article("1", "Council met today", before),
        _article("2", positive, after),
        _article("3", negative, after, source="Source B"),
        _article("4", "Council met today", after, source="Source A"),
        _article("5", positive, after, is_relevant=False),
        _article("6", positive, after, region_status="out_of_region"),
        _article("7", positive, after, is_synthetic=True),
        _article("8", positive, after, dataset_kind="demo"),
        _article("9", positive, after, event_time=None),
    ]
    frame = spark.createDataFrame(rows, article_schema()).repartition(2)
    result = {row.day: row.asDict() for row in summarize_articles(frame, kind="live").collect()}
    assert set(result) == {"2026-10-08", "2026-10-09"}
    assert result["2026-10-08"]["article_count"] == 1
    day = result["2026-10-09"]
    assert day["article_count"] == 3
    assert day["positive_count"] == day["negative_count"] == day["neutral_count"] == 1
    assert day["source_count"] == 2
    assert day["mean_compound"] == pytest.approx((score_headline(positive)[0] + score_headline(negative)[0]) / 3)


def test_empty_input_and_demo_synthetic_data(spark):
    empty = spark.createDataFrame([], article_schema())
    summary = summarize_articles(empty, kind="live")
    assert summary.columns == SUMMARY_COLUMNS
    assert summary.count() == 0
    when = datetime(2026, 10, 9, tzinfo=timezone.utc)
    demo = spark.createDataFrame([
        _article("demo", "Wonderful success", when, dataset_kind="demo", is_synthetic=True, source=None),
        _article("live", "Terrible disaster", when),
    ], article_schema()).repartition(2)
    result = summarize_articles(demo, kind="demo").collect()
    assert len(result) == 1
    assert result[0].article_count == result[0].positive_count == result[0].source_count == 1
