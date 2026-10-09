"""Run the ingestion pipeline once (manual / debugging).

    python scripts/run_pipeline.py                     # RSS + Ollama (fallback if down)
    python scripts/run_pipeline.py --no-llm            # force keyword fallback
    python scripts/run_pipeline.py --reclassify-fallbacks   # retry fallback rows with the LLM
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from voterpulse import database as db
from voterpulse.config import get_config
from voterpulse.logging_utils import setup_from_config
from voterpulse.monitoring.pipeline import reclassify_fallbacks, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-llm", action="store_true", help="skip Ollama and use the keyword classifier")
    parser.add_argument("--reclassify-fallbacks", action="store_true",
                        help="re-queue keyword-fallback articles (LLM was down or answered invalidly) for the LLM")
    args = parser.parse_args()
    setup_from_config()
    cfg = get_config()
    if args.reclassify_fallbacks:
        conn = db.init_db(cfg.live_db_path, "live")
        reclassify_fallbacks(conn)
        conn.close()
    report = run_pipeline(cfg, trigger="manual", llm=not args.no_llm)
    s = report.stats
    print(f"\nRun {report.run_id}: {report.status.upper()}")
    print(f"  feeds ok/failed:      {s.feeds_ok}/{s.feeds_failed}")
    print(f"  headlines collected:  {s.headlines_collected}  (new {s.new_headlines}, duplicates {s.duplicates})")
    print(f"  classified:           {s.classified}  (LLM {s.llm_classified}, keyword fallback {s.fallback_classified})")
    print(f"  rejected (not election issue): {s.rejected}   failed: {s.failed}")
    print(f"  still waiting for classification: {report.backlog}")
    print(f"  duration:             {s.duration_seconds}s")
    if report.llm_note:
        print(f"  NOTE: {report.llm_note}")
    for key, err in report.feed_errors.items():
        print(f"  feed {key} failed: {err}")
    return 0 if report.status in {"success", "partial", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main())
