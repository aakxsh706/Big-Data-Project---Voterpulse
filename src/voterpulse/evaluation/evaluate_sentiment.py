"""Compare VADER labels with three human annotators.

Gold label = strict majority of the three annotators (at least 2 of 3 agree). Items
where all three disagree are excluded from accuracy/F1 and reported separately; they
should be discussed by the annotators and either resolved by consensus (recorded in
``sentiment_notes``) or left out. Agreement is reported as Fleiss' kappa (all three)
and pairwise Cohen's kappa.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from voterpulse.config import AppConfig
from voterpulse.evaluation import metrics as m
from voterpulse.evaluation.sampling import read_annotations
from voterpulse.sentiment.vader_analyzer import VaderAnalyzer

LABELS = ["Positive", "Neutral", "Negative"]
ANNOTATORS = ["sentiment_annotator_1", "sentiment_annotator_2", "sentiment_annotator_3"]


def _norm(value: str) -> str:
    value = value.strip().capitalize()
    return value if value in LABELS else ""


def evaluate(cfg: AppConfig, input_path: Path, out_dir: Path) -> dict:
    rows = read_annotations(input_path)
    complete, partial = [], 0
    for row in rows:
        labels = [_norm(row.get(col, "")) for col in ANNOTATORS]
        if all(labels):
            complete.append((row, labels))
        elif any(labels):
            partial += 1
    if not complete:
        raise SystemExit(f"No fully annotated rows in {input_path}. All three sentiment columns must be filled.")

    vader = VaderAnalyzer(cfg.thresholds.headline)
    ratings = [labels for _, labels in complete]
    fleiss = m.fleiss_kappa(ratings, LABELS)
    pairwise = {f"{a[-1]}-{b[-1]}": m.cohen_kappa([r[i] for r in ratings], [r[j] for r in ratings])
                for (i, a), (j, b) in combinations(enumerate(ANNOTATORS), 2)}

    gold, pred, unresolved, details = [], [], [], []
    for row, labels in complete:
        majority = m.majority_label(labels)
        scores = vader.score(row["headline"])
        details.append({"item_id": row["item_id"], "headline": row["headline"], "annotators": "|".join(labels),
                        "gold": majority or "", "vader": scores.label.value, "compound": scores.compound})
        if majority is None:
            unresolved.append(row["item_id"])
            continue
        gold.append(majority)
        pred.append(scores.label.value)

    per = m.per_label(gold, pred, LABELS)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(input_path),
        "items_fully_annotated": len(complete),
        "items_partially_annotated_skipped": partial,
        "items_without_majority": len(unresolved),
        "unresolved_item_ids": unresolved,
        "fleiss_kappa": fleiss,
        "pairwise_cohen_kappa": pairwise,
        "vader_accuracy": m.accuracy(gold, pred),
        "vader_macro_f1": m.macro([r.f1 for r in per]),
        "per_label": [r.__dict__ for r in per],
        "confusion_labels": LABELS,
        "confusion": m.confusion_matrix(gold, pred, LABELS),
        "thresholds": cfg.thresholds.headline.model_dump(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sentiment_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "sentiment_predictions.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(details[0]))
        w.writeheader()
        w.writerows(details)
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [f"# Sentiment evaluation — {result['timestamp']}", "",
             f"- Fully annotated: {len(complete)} · without majority (excluded): {len(unresolved)} · "
             f"partially annotated (skipped): {partial}",
             f"- Fleiss' kappa (3 annotators): {fmt(fleiss)}",
             "- Pairwise Cohen's kappa: " + ", ".join(f"{k}: {fmt(v)}" for k, v in pairwise.items()),
             f"- VADER accuracy vs majority: {fmt(result['vader_accuracy'])} · macro F1: {fmt(result['vader_macro_f1'])}",
             "", "| gold \\ VADER | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    for label, line in zip(LABELS, result["confusion"]):
        lines.append(f"| {label} | " + " | ".join(str(x) for x in line) + " |")
    (out_dir / "sentiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
