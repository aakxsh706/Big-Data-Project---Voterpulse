"""Evaluation toolkit. All metrics are computed from YOUR annotations; none are bundled.

    python scripts/evaluate.py sample --n 300                       # export headlines to annotate
    python scripts/evaluate.py classifier data/evaluation/annotated.csv --method both --repeat 3
    python scripts/evaluate.py sentiment  data/evaluation/annotated.csv
    python scripts/evaluate.py benchmark  --n 50 --cpu-only

Results are written to data/evaluation/results/<timestamp>/.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from voterpulse import database as db
from voterpulse.config import get_config
from voterpulse.logging_utils import setup_from_config


def results_dir(cfg, kind: str) -> Path:
    return cfg.resolve_path(f"data/evaluation/results/{datetime.now():%Y%m%d_%H%M%S}_{kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="export a stratified sample of live headlines for annotation")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--out", default="data/evaluation/to_annotate.csv")
    c = sub.add_parser("classifier", help="LLM vs keyword baseline against gold issue labels")
    c.add_argument("input")
    c.add_argument("--method", choices=["llm", "keyword", "both"], default="both")
    c.add_argument("--repeat", type=int, default=1, help="repeat LLM runs to measure reproducibility")
    t = sub.add_parser("sentiment", help="VADER vs three human annotators")
    t.add_argument("input")
    b = sub.add_parser("benchmark", help="latency / memory / CPU-only benchmark of the LLM")
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--input", help="CSV with a 'headline' column (default: latest live headlines)")
    b.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    setup_from_config()
    cfg = get_config()
    if args.command == "sample":
        conn = db.init_db(cfg.live_db_path, "live")
        n = __import__("voterpulse.evaluation.sampling", fromlist=["x"]).export_sample(
            conn, cfg, cfg.resolve_path(args.out), args.n, args.seed)
        print(f"Wrote {n} headlines to {args.out}. Fill gold_issue (and the 3 sentiment columns), "
              "then save as data/evaluation/annotated.csv.")
        if n < args.n:
            print(f"Only {n} live headlines available; let the scheduler collect more before sampling.")
    elif args.command == "classifier":
        from voterpulse.evaluation.evaluate_classifier import evaluate
        out = results_dir(cfg, "classifier")
        evaluate(cfg, Path(args.input), out, args.method, args.repeat)
        print((out / "report.md").read_text(encoding="utf-8"))
    elif args.command == "sentiment":
        from voterpulse.evaluation.evaluate_sentiment import evaluate
        out = results_dir(cfg, "sentiment")
        evaluate(cfg, Path(args.input), out)
        print((out / "sentiment_report.md").read_text(encoding="utf-8"))
    elif args.command == "benchmark":
        from voterpulse.evaluation.benchmark import run_benchmark
        from voterpulse.evaluation.sampling import read_annotations
        if args.input:
            headlines = [r["headline"] for r in read_annotations(Path(args.input))][: args.n]
        else:
            conn = db.init_db(cfg.live_db_path, "live")
            headlines = [r[0] for r in conn.execute(
                "SELECT headline FROM articles ORDER BY ingested_at DESC LIMIT ?", (args.n,))]
        if not headlines:
            raise SystemExit("No headlines to benchmark. Run the pipeline first or pass --input.")
        out = results_dir(cfg, "benchmark")
        print(json.dumps(run_benchmark(cfg, headlines, out, args.cpu_only), indent=2))


if __name__ == "__main__":
    from voterpulse.classification.llm_classifier import OllamaUnavailable

    try:
        main()
    except OllamaUnavailable as exc:
        raise SystemExit(f"Ollama is not ready: {exc}\nStart Ollama and run `ollama pull llama3.2:3b`, "
                         "or use --method keyword for the classifier evaluation.") from None
