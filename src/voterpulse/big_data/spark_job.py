"""Parallel MongoDB -> Spark VADER scoring -> versioned daily summaries.

The article corpus stays on Spark executors. Only scalar counts and run metadata
return to the driver. A run becomes visible to readers only after its complete
summary has been written; failed writes never replace the last successful run.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from voterpulse.config import AppConfig


_ANALYZER: Any = None
DAY_TIME_ZONE = "Asia/Kolkata"
SUMMARY_COLUMNS = [
    "day", "issue", "article_count", "positive_count", "neutral_count",
    "negative_count", "mean_compound", "source_count",
]


class SparkJobError(RuntimeError):
    """A safe, actionable failure message for the command-line runner."""


def score_headline(text: str | None, positive_min: float = 0.05,
                   negative_max: float = -0.05) -> tuple[float, str]:
    """Initialize VADER once per Python worker and use application thresholds."""
    global _ANALYZER
    if _ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _ANALYZER = SentimentIntensityAnalyzer()
    compound = max(-1.0, min(1.0, float(_ANALYZER.polarity_scores(text or "")["compound"])))
    label = "Positive" if compound >= positive_min else (
        "Negative" if compound <= negative_max else "Neutral"
    )
    return compound, label


def article_schema():
    """Explicit schema also allows Spark to process an empty Mongo collection."""
    from pyspark.sql.types import BooleanType, StringType, StructField, StructType, TimestampType

    return StructType([
        StructField(name, StringType(), True)
        for name in ("_id", "headline", "normalized_headline", "source", "url", "issue",
                     "region_status", "dataset_kind")
    ] + [
        StructField(name, TimestampType(), True)
        for name in ("published_at", "ingested_at", "event_time")
    ] + [
        StructField("is_relevant", BooleanType(), True),
        StructField("is_synthetic", BooleanType(), True),
    ])


def eligible_articles(frame: DataFrame, kind: str) -> DataFrame:
    """Keep accepted primary classifications in the selected dataset and region."""
    from pyspark.sql import functions as F

    if kind not in {"live", "demo"}:
        raise ValueError("kind must be 'live' or 'demo'")
    eligible = frame.filter(
        (F.col("dataset_kind") == kind)
        & (F.col("is_relevant") == F.lit(True))
        & (F.coalesce(F.col("region_status"), F.lit("")) != "out_of_region")
        & F.col("event_time").isNotNull()
        & F.col("issue").isNotNull()
        & (F.length(F.trim(F.col("issue"))) > 0)
    )
    if kind == "live":
        eligible = eligible.filter(~F.coalesce(F.col("is_synthetic"), F.lit(False)))
    return eligible


def summarize_articles(frame: DataFrame, *, kind: str, positive_min: float = 0.05,
                       negative_max: float = -0.05) -> DataFrame:
    """Score headlines on workers and aggregate primary issues by IST calendar day."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    score_type = StructType([
        StructField("compound", DoubleType(), False),
        StructField("label", StringType(), False),
    ])
    score = F.udf(lambda headline: score_headline(headline, positive_min, negative_max), score_type)
    scored = eligible_articles(frame, kind).withColumn("sentiment", score("headline"))
    scored = scored.withColumn(
        "day", F.date_format(F.from_utc_timestamp("event_time", DAY_TIME_ZONE), "yyyy-MM-dd")
    ).withColumn(
        "source_name", F.when(F.col("source").isNull() | (F.trim("source") == ""),
                              F.lit("Unknown source")).otherwise(F.col("source"))
    )
    result = scored.groupBy("day", "issue").agg(
        F.count(F.lit(1)).alias("article_count"),
        F.sum(F.when(F.col("sentiment.label") == "Positive", 1).otherwise(0)).alias("positive_count"),
        F.sum(F.when(F.col("sentiment.label") == "Neutral", 1).otherwise(0)).alias("neutral_count"),
        F.sum(F.when(F.col("sentiment.label") == "Negative", 1).otherwise(0)).alias("negative_count"),
        F.avg("sentiment.compound").alias("mean_compound"),
        F.countDistinct("source_name").alias("source_count"),
    )
    return result.select(*SUMMARY_COLUMNS)


def connector_uri(uri: str) -> str:
    """Keep explicit database isolation and acknowledged-write settings authoritative.

    Mongo's Spark connector gives URI database paths priority over the database
    option. Retain that path as authSource when needed, then remove the path.
    URI write-concern options also take precedence, so remove them before the
    writer explicitly requests majority acknowledgement with a bounded timeout.
    """
    parts = urlsplit(uri)
    write_options = {"w", "wtimeoutms", "wtimeout", "writeconcern.w", "writeconcern.wtimeoutms"}
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if key.lower() not in write_options]
    database = unquote(parts.path.lstrip("/"))
    if database and not any(key.lower() == "authsource" for key, _ in query):
        query.append(("authSource", database))
    return urlunsplit((parts.scheme, parts.netloc, "/", urlencode(query), ""))


def redact_error(error: Exception, uri: str = "") -> str:
    """Avoid persisting Mongo credentials from connector/driver exceptions."""
    message = str(error)
    if uri:
        message = message.replace(uri, "[MongoDB URI redacted]")
        parts = urlsplit(uri)
        for value in (parts.password, parts.username):
            if value:
                message = message.replace(value, "[redacted]").replace(unquote(value), "[redacted]")
    return re.sub(r"mongodb(?:\+srv)?://[^\s'\"<>]+", "[MongoDB URI redacted]", message)[:1500]


def _package_workers(directory: str) -> Path:
    """Ship this module to workers even when the source project isn't installed."""
    package_root = Path(__file__).resolve().parents[1]
    archive = Path(directory) / "voterpulse_workers.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for source in package_root.rglob("*.py"):
            bundle.write(source, source.relative_to(package_root.parent))
    return archive


def create_spark_session(cfg: AppConfig) -> SparkSession:
    """Create the dedicated batch-job session with the MongoDB connector."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise SparkJobError("Install the big-data extra: python -m pip install -e '.[big-data]'") from exc
    settings = cfg.settings.big_data
    builder = (SparkSession.builder.appName("VoterPulse MongoDB daily sentiment")
               .master(settings.spark_master)
               .config("spark.jars.packages", settings.spark_connector)
               .config("spark.jars.ivy", str(cfg.resolve_path("data/spark/ivy")))
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.shuffle.partitions", settings.spark_partitions)
               .config("spark.default.parallelism", settings.spark_partitions)
               .config("spark.redaction.regex", "(?i)secret|password|token|access.key|connection.uri")
               .config("spark.redaction.string.regex", r"mongodb(?:\+srv)?://[^\s'\"]+"))
    if settings.spark_master.startswith("local"):
        builder = builder.config("spark.pyspark.python", sys.executable)
    spark = builder.getOrCreate()
    if settings.spark_master.startswith("local"):
        spark.sparkContext.pythonExec = sys.executable
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _read_articles(spark: SparkSession, uri: str, database_name: str, kind: str) -> DataFrame:
    return (spark.read.format("mongodb").schema(article_schema())
            .option("connection.uri", uri)
            .option("database", database_name).option("collection", "articles")
            .option("partitioner", "com.mongodb.spark.sql.connector.read.partitioner.SamplePartitioner")
            .option("partitioner.options.partition.field", "_id")
            .option("aggregation.pipeline", json.dumps([{"$match": {"dataset_kind": kind}}]))
            .load())


def _write_summaries(frame: DataFrame, uri: str, database_name: str, timeout_ms: int) -> None:
    (frame.write.format("mongodb").mode("append")
     .option("connection.uri", uri)
     .option("database", database_name).option("collection", "daily_summaries")
     .option("operationType", "replace").option("idFieldList", "_id")
     .option("writeConcern.w", "majority").option("writeConcern.wTimeoutMS", timeout_ms)
     .option("upsertDocument", "true").save())


def run_spark_job(cfg: AppConfig, kind: str = "live", *, explain: bool = False) -> dict[str, Any]:
    """Publish an isolated successful run without collecting articles to Python."""
    from voterpulse.big_data.mongo_store import open_mongo

    if kind not in {"live", "demo"}:
        raise ValueError("kind must be 'live' or 'demo'")
    settings = cfg.settings.big_data
    raw_uri = settings.mongo_uri.get_secret_value()
    uri = connector_uri(raw_uri)
    name = settings.mongo_demo_database if kind == "demo" else settings.mongo_database
    run_id = uuid4().hex
    start = time.monotonic()
    metadata: dict[str, Any] = {
        "_id": run_id, "run_id": run_id, "dataset_kind": kind, "status": "running",
        "started_at": datetime.now(timezone.utc), "spark_master": settings.spark_master,
        "config": {"positive_min": cfg.thresholds.headline.positive_min,
                   "negative_max": cfg.thresholds.headline.negative_max,
                   "time_zone": DAY_TIME_ZONE, "connector": settings.spark_connector},
    }
    spark = None
    articles = summaries = None
    with open_mongo(cfg, kind, initialize=True) as database, tempfile.TemporaryDirectory(prefix="voterpulse-spark-") as work:
        database.spark_runs.insert_one(metadata.copy())
        try:
            from pyspark import StorageLevel
            from pyspark.sql import functions as F

            spark = create_spark_session(cfg)
            spark.sparkContext.addPyFile(str(_package_workers(work)))
            articles = (_read_articles(spark, uri, name, kind)
                        .repartition(settings.spark_partitions).persist(StorageLevel.MEMORY_AND_DISK))
            metadata.update(input_count=articles.count(),
                            eligible_count=eligible_articles(articles, kind).count(),
                            partitions=articles.rdd.getNumPartitions(), spark_version=spark.version)
            summaries = summarize_articles(
                articles, kind=kind, positive_min=cfg.thresholds.headline.positive_min,
                negative_max=cfg.thresholds.headline.negative_max,
            ).persist(StorageLevel.MEMORY_AND_DISK)
            if explain:
                summaries.explain(mode="formatted")
            metadata["summary_count"] = summaries.count()
            if metadata["summary_count"]:
                output = (summaries.withColumn("run_id", F.lit(run_id))
                          .withColumn("dataset_kind", F.lit(kind))
                          .withColumn("_id", F.sha2(F.to_json(F.struct("run_id", "day", "issue")), 256)))
                _write_summaries(output, uri, name, settings.mongo_timeout_ms)
            metadata.update(status="success", completed_at=datetime.now(timezone.utc),
                            duration_seconds=round(time.monotonic() - start, 3))
            database.spark_runs.replace_one({"_id": run_id}, metadata)
            return metadata
        except Exception as exc:
            error = redact_error(exc, raw_uri)
            metadata.update(status="failed", completed_at=datetime.now(timezone.utc),
                            duration_seconds=round(time.monotonic() - start, 3),
                            error_type=type(exc).__name__, error=error)
            try:
                database.spark_runs.replace_one({"_id": run_id}, metadata)
            except Exception:
                pass  # An unavailable MongoDB must not hide the original Spark error.
            if isinstance(exc, ImportError):
                error = "Install the big-data extra: python -m pip install -e '.[big-data]'"
            raise SparkJobError(f"Spark run failed ({type(exc).__name__}): {error}") from None
        finally:
            for frame in (summaries, articles):
                if frame is not None:
                    try:
                        frame.unpersist()
                    except Exception:
                        pass
            if spark is not None:
                try:
                    spark.stop()
                except Exception:
                    pass
