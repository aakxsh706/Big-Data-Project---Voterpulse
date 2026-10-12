"""Sync articles to MongoDB, then score and aggregate them with Apache Spark.

    python scripts/run_big_data.py --demo
    python scripts/run_big_data.py --sync-only
    python scripts/run_big_data.py --skip-sync --explain

Without --demo this command always uses the live dataset.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from voterpulse import database as db
from voterpulse.big_data.mongo_store import sync_articles
from voterpulse.big_data.spark_job import redact_error, run_spark_job
from voterpulse.config import get_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="use the isolated synthetic demo dataset")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync-only", action="store_true", help="sync articles to MongoDB without starting Spark")
    mode.add_argument("--skip-sync", action="store_true", help="process the articles already stored in MongoDB")
    parser.add_argument("--full-sync", action="store_true", help="resend every article to an empty/new MongoDB destination (does not prune extras)")
    parser.add_argument("--explain", action="store_true", help="print Spark's physical aggregation plan")
    args = parser.parse_args(argv)
    if args.full_sync and args.skip_sync:
        parser.error("--full-sync cannot be used with --skip-sync")
    cfg = get_config()
    kind = "demo" if args.demo else "live"
    try:
        if not args.skip_sync:
            path = cfg.demo_db_path if args.demo else cfg.live_db_path
            if not path.is_file():
                raise RuntimeError(f"Source database does not exist: {path}. Run the ingestion or demo generator first.")
            conn = db.init_db(path, kind, cfg.settings.database.busy_timeout_ms)
            try:
                if db.dataset_kind(conn) != kind:
                    raise RuntimeError(f"Source database is not marked as {kind!r}; refusing to mix datasets.")
                count = sync_articles(conn, cfg, kind, full=args.full_sync)
            finally:
                conn.close()
            print(f"Synced {count:,} {kind} article changes to MongoDB.")
        if args.sync_only:
            return 0
        result = run_spark_job(cfg, kind, explain=args.explain)
        print(f"Spark run {result['run_id']}: SUCCESS ({kind})")
        print(f"  Articles read / eligible: {result['input_count']:,} / {result['eligible_count']:,}")
        print(f"  Daily issue summaries:   {result['summary_count']:,} (Asia/Kolkata calendar days)")
        print(f"  Spark / partitions:      {result['spark_version']} / {result['partitions']}")
        print(f"  Duration:                {result['duration_seconds']}s")
        return 0
    except Exception as exc:
        print(f"Big-data job failed: {redact_error(exc, cfg.settings.big_data.mongo_uri.get_secret_value())}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
