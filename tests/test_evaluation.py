"""Tests for evaluation tooling. Fixture rows are unit-test inputs, not research data."""

import csv
import json

import pytest

from conftest import FakeSession, ollama_answer, ollama_tags, seed, NOW
from voterpulse.classification.llm_classifier import OllamaClassifier
from voterpulse.evaluation import metrics as m
from voterpulse.evaluation.evaluate_classifier import evaluate as eval_classifier
from voterpulse.evaluation.evaluate_sentiment import evaluate as eval_sentiment
from voterpulse.evaluation.sampling import TEMPLATE_COLUMNS, export_sample


def test_per_label_and_macro():
    gold = ["A", "A", "B", "B", "C"]
    pred = ["A", "B", "B", "B", "A"]
    rows = {r.label: r for r in m.per_label(gold, pred, ["A", "B", "C"])}
    assert rows["A"].precision == 0.5 and rows["A"].recall == 0.5
    assert rows["B"].precision == pytest.approx(2 / 3) and rows["B"].recall == 1.0
    assert rows["C"].precision is None and rows["C"].recall == 0.0   # never predicted -> undefined
    assert m.macro([0.5, None, 1.0]) == 0.75
    assert m.confusion_matrix(gold, pred, ["A", "B", "C"]) == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_kappas_against_hand_computed_values():
    assert m.cohen_kappa(list("YYNN"), list("YNNN")) == pytest.approx(0.5)
    assert m.fleiss_kappa([["A", "A", "B"], ["A", "B", "B"]], ["A", "B"]) == pytest.approx(-1 / 3)
    assert m.fleiss_kappa([["A"] * 3, ["B"] * 3], ["A", "B"]) == pytest.approx(1.0)


def test_majority_label():
    assert m.majority_label(["Positive", "Positive", "Negative"]) == "Positive"
    assert m.majority_label(["Positive", "Neutral", "Negative"]) is None


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=TEMPLATE_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in TEMPLATE_COLUMNS})


def test_classifier_evaluation_keyword_and_llm_with_repeat(cfg, tmp_path):
    data = tmp_path / "annotated.csv"
    _write(data, [
        {"item_id": "T1", "headline": "MSP hike fails to calm mandi protests", "gold_issue": "Farmers and Agriculture"},
        {"item_id": "T2", "headline": "Power struggle in state cabinet", "gold_issue": "Not an election issue"},
        {"item_id": "T3", "headline": "Youth unemployment rises", "gold_issue": "Jobs and Unemployment"},
        {"item_id": "T4", "headline": "Unlabelled headline"},
        {"item_id": "T5", "headline": "Bad label", "gold_issue": "Weather"},
    ])
    calls = {"n": 0}

    def post(url, payload, **kw):
        calls["n"] += 1
        text = payload["messages"][-1]["content"]
        if "MSP" in text:
            return ollama_answer("Farmers and Agriculture", "MSP hike")
        if "Power" in text:
            # flips between runs to exercise the reproducibility report
            return ollama_answer("Not an election issue" if calls["n"] < 4 else "Electricity and Utilities", "Power")
        return ollama_answer("Jobs and Unemployment", "Youth unemployment")

    client = OllamaClassifier.from_config(cfg, session=FakeSession(get=lambda u, **k: ollama_tags(), post=post),
                                          sleep=lambda s: None)
    out = tmp_path / "res"
    result = eval_classifier(cfg, data, out, "both", repeat=2, client=client)
    assert result["meta"]["n"] == 3 and result["meta"]["skipped"] == 1 and len(result["meta"]["problems"]) == 1
    by = {s["method"]: s for s in result["summaries"]}
    assert by["keyword"]["accuracy"] == 1.0 and by["llm"]["accuracy"] == 1.0
    assert by["llm"]["rejection_recall"] == 1.0
    assert result["reproducibility"]["identical_rate"] == pytest.approx(2 / 3)
    for name in ("report.md", "metrics.json", "predictions.csv", "confusion_llm.csv", "per_label_keyword.csv"):
        assert (out / name).exists()
    json.loads((out / "metrics.json").read_text())


def test_sentiment_evaluation(cfg, tmp_path):
    data = tmp_path / "sent.csv"
    _write(data, [
        {"item_id": "S1", "headline": "Farmers angry as crop prices crash",
         "sentiment_annotator_1": "Negative", "sentiment_annotator_2": "negative", "sentiment_annotator_3": "Neutral"},
        {"item_id": "S2", "headline": "New hospital brings great relief and hope",
         "sentiment_annotator_1": "Positive", "sentiment_annotator_2": "Positive", "sentiment_annotator_3": "Positive"},
        {"item_id": "S3", "headline": "Parliament session begins",
         "sentiment_annotator_1": "Neutral", "sentiment_annotator_2": "Positive", "sentiment_annotator_3": "Negative"},
        {"item_id": "S4", "headline": "Half done", "sentiment_annotator_1": "Neutral"},
    ])
    r = eval_sentiment(cfg, data, tmp_path / "out")
    assert r["items_fully_annotated"] == 3 and r["items_without_majority"] == 1
    assert r["items_partially_annotated_skipped"] == 1
    assert r["vader_accuracy"] == 1.0
    assert (tmp_path / "out" / "sentiment_report.md").exists()


def test_export_sample_is_stratified_and_blank(cfg, conn, tmp_path):
    seed(conn, "Jobs and Unemployment", 0.0, NOW, n=10)
    out = tmp_path / "to_annotate.csv"
    assert export_sample(conn, cfg, out, n=5) == 5
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 5 and all(r["gold_issue"] == "" for r in rows)
