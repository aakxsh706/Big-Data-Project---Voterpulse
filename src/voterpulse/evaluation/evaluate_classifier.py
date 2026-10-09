"""Evaluate the local LLM and the keyword baseline on human-annotated headlines.

Input: the annotation CSV (``gold_issue`` filled by people). Output (in ``--out``):
predictions.csv, per_label_<method>.csv, confusion_<method>.csv, metrics.json and
report.md. Rows without a gold label are skipped and counted, never guessed.

Reproducibility: ``--repeat K`` classifies every headline K times with the LLM
(same model, prompt, settings) and reports any label that differs between runs.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from voterpulse.classification.keyword_classifier import KeywordClassifier
from voterpulse.classification.llm_classifier import OllamaClassifier, OllamaUnavailable
from voterpulse.classification.validator import LLMResponseError
from voterpulse.config import AppConfig
from voterpulse.evaluation import metrics as m
from voterpulse.evaluation.sampling import read_annotations
from voterpulse.logging_utils import fields, get_logger

log = get_logger(__name__)

ERROR_LABEL = "<error>"
TARGET_PRECISION = 0.90
TARGET_RECALL = 0.85


@dataclass
class MethodResult:
    method: str
    predictions: list[str]
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0


def load_gold(path: Path, cfg: AppConfig) -> tuple[list[dict[str, str]], int, list[str]]:
    """Return (usable rows, skipped_unlabelled, problems). Unknown labels are reported, not fixed."""
    rows = read_annotations(path)
    allowed = set(cfg.issues.all_labels)
    usable, skipped, problems = [], 0, []
    for row in rows:
        label = row.get("gold_issue", "")
        if not label:
            skipped += 1
        elif label not in allowed:
            problems.append(f"{row['item_id']}: unknown gold_issue {label!r}")
        else:
            usable.append(row)
    return usable, skipped, problems


def run_keyword(cfg: AppConfig, headlines: list[str]) -> MethodResult:
    kc = KeywordClassifier(cfg.issues)
    result = MethodResult("keyword", [])
    for text in headlines:
        start = time.perf_counter()
        result.predictions.append(kc.classify(text).issue)
        result.latencies_ms.append((time.perf_counter() - start) * 1000)
    return result


def run_llm(cfg: AppConfig, headlines: list[str], client: OllamaClassifier | None = None,
            label: str = "llm") -> MethodResult:
    """Classify with the LLM directly (no cache, no fallback). Failures count as errors."""
    client = client or OllamaClassifier.from_config(cfg)
    client.resolve_model_id()  # fail fast with a clear message if Ollama is not ready
    result = MethodResult(label, [])
    for i, text in enumerate(headlines, start=1):
        try:
            out = client.classify(text)
            result.predictions.append(out.parsed.issue)
            result.latencies_ms.append(out.latency_ms)
        except (OllamaUnavailable, LLMResponseError) as exc:
            result.predictions.append(ERROR_LABEL)
            result.errors += 1
            log.warning("eval_llm_error", extra=fields(item=i, error=str(exc)[:160]))
        if i % 25 == 0:
            log.info("eval_progress", extra=fields(method=label, done=i, total=len(headlines)))
    return result


def summarise(cfg: AppConfig, gold: list[str], result: MethodResult) -> dict:
    issues = cfg.issues.issue_names
    labels = cfg.issues.all_labels
    rows = m.per_label(gold, result.predictions, labels)
    not_rel = cfg.issues.not_relevant_label
    gold_irrelevant = [i for i, g in enumerate(gold) if g == not_rel]
    pred_irrelevant = [i for i, p in enumerate(result.predictions) if p == not_rel]
    issue_rows = [r for r in rows if r.label in issues]
    lat = sorted(result.latencies_ms)
    return {
        "method": result.method,
        "n": len(gold),
        "errors": result.errors,
        "accuracy": m.accuracy(gold, result.predictions),
        "macro_precision_6_issues": m.macro([r.precision for r in issue_rows]),
        "macro_recall_6_issues": m.macro([r.recall for r in issue_rows]),
        "macro_f1_6_issues": m.macro([r.f1 for r in issue_rows]),
        "undefined_metrics": [r.label for r in issue_rows if r.precision is None or r.recall is None],
        "rejection_recall": (sum(result.predictions[i] == not_rel for i in gold_irrelevant) / len(gold_irrelevant)
                             if gold_irrelevant else None),
        "rejection_precision": (sum(gold[i] == not_rel for i in pred_irrelevant) / len(pred_irrelevant)
                                if pred_irrelevant else None),
        "avg_latency_ms": sum(lat) / len(lat) if lat else None,
        "p95_latency_ms": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "per_label": [asdict(r) | {
            "meets_precision_target": None if r.precision is None else r.precision >= TARGET_PRECISION,
            "meets_recall_target": None if r.recall is None else r.recall >= TARGET_RECALL,
        } for r in rows],
        "confusion_labels": labels,
        "confusion": m.confusion_matrix(gold, result.predictions, labels),
    }


def reproducibility(runs: list[list[str]], headlines: list[str]) -> dict:
    differing = [{"headline": headlines[i], "labels": [run[i] for run in runs]}
                 for i in range(len(headlines)) if len({run[i] for run in runs}) > 1]
    return {"runs": len(runs), "items": len(headlines),
            "identical_rate": 1 - len(differing) / len(headlines) if headlines else None,
            "differing": differing}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def write_outputs(out: Path, cfg: AppConfig, rows: list[dict[str, str]], summaries: list[dict],
                  results: list[MethodResult], meta: dict, repro: dict | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["item_id", "headline", "gold_issue"] + [r.method for r in results])
        for i, row in enumerate(rows):
            w.writerow([row["item_id"], row["headline"], row["gold_issue"]] + [r.predictions[i] for r in results])
    for s in summaries:
        with (out / f"per_label_{s['method']}.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(s["per_label"][0]))
            w.writeheader()
            w.writerows(s["per_label"])
        with (out / f"confusion_{s['method']}.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["gold \\ predicted"] + s["confusion_labels"])
            for label, line in zip(s["confusion_labels"], s["confusion"]):
                w.writerow([label] + line)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "methods": summaries, "reproducibility": repro},
                                                 indent=2), encoding="utf-8")
    lines = [f"# Classifier evaluation — {meta['timestamp']}", "",
             f"- Annotated file: `{meta['input']}`",
             f"- Headlines evaluated: {meta['n']} (skipped without gold label: {meta['skipped']}, "
             f"invalid gold labels: {len(meta['problems'])})",
             f"- Model: `{meta.get('model_id') or 'not used'}` · prompt version `{meta['prompt_version']}`",
             f"- Targets: precision ≥ {TARGET_PRECISION}, recall ≥ {TARGET_RECALL} per issue "
             "(objectives, not guarantees)", ""]
    for s in summaries:
        lines += [f"## {s['method']}", "",
                  f"Accuracy {_fmt(s['accuracy'])} · macro P {_fmt(s['macro_precision_6_issues'])} · "
                  f"macro R {_fmt(s['macro_recall_6_issues'])} · macro F1 {_fmt(s['macro_f1_6_issues'])} · "
                  f"errors {s['errors']}", "",
                  f"Rejecting irrelevant headlines: recall {_fmt(s['rejection_recall'])}, "
                  f"precision {_fmt(s['rejection_precision'])} · avg time {_fmt(s['avg_latency_ms'])} ms", "",
                  "| Label | Precision | Recall | F1 | Support | P target | R target |", "|---|---|---|---|---|---|---|"]
        for r in s["per_label"]:
            tick = lambda v: "n/a" if v is None else ("yes" if v else "no")  # noqa: E731
            lines.append(f"| {r['label']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | "
                         f"{r['support']} | {tick(r['meets_precision_target'])} | {tick(r['meets_recall_target'])} |")
        lines.append("")
    if "agreement_llm_keyword" in meta:
        lines += [f"LLM vs keyword agreement on this set: {_fmt(meta['agreement_llm_keyword'])}", ""]
    if repro:
        lines += ["## Reproducibility", "",
                  f"{repro['runs']} repeated LLM runs: identical labels for {_fmt(repro['identical_rate'])} of items; "
                  f"{len(repro['differing'])} item(s) differed (listed in metrics.json).", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(cfg: AppConfig, input_path: Path, out_dir: Path, method: str = "both", repeat: int = 1,
             client: OllamaClassifier | None = None) -> dict:
    rows, skipped, problems = load_gold(input_path, cfg)
    if not rows:
        raise SystemExit(f"No annotated rows in {input_path}. Fill the gold_issue column first.")
    headlines = [r["headline"] for r in rows]
    gold = [r["gold_issue"] for r in rows]
    results: list[MethodResult] = []
    meta: dict = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "input": str(input_path), "n": len(rows), "skipped": skipped, "problems": problems}
    if method in ("keyword", "both"):
        results.append(run_keyword(cfg, headlines))
    repro = None
    llm_client = client
    if method in ("llm", "both"):
        llm_client = llm_client or OllamaClassifier.from_config(cfg)
        runs = [run_llm(cfg, headlines, llm_client, "llm")]
        for k in range(2, repeat + 1):
            runs.append(run_llm(cfg, headlines, llm_client, f"llm_run{k}"))
        results.append(runs[0])
        meta["model_id"] = llm_client.resolve_model_id()
        if repeat > 1:
            repro = reproducibility([r.predictions for r in runs], headlines)
    meta["prompt_version"] = (llm_client or OllamaClassifier.from_config(cfg)).prompt_version
    summaries = [summarise(cfg, gold, r) for r in results]
    by = {r.method: r.predictions for r in results}
    if "llm" in by and "keyword" in by:
        meta["agreement_llm_keyword"] = m.accuracy(by["keyword"], by["llm"])
    write_outputs(out_dir, cfg, rows, summaries, results, meta, repro)
    return {"meta": meta, "summaries": summaries, "reproducibility": repro}
