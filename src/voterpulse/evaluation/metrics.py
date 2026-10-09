"""Dependency-free classification metrics and inter-annotator agreement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class LabelMetrics:
    label: str
    precision: float | None   # None when the label was never predicted (undefined)
    recall: float | None      # None when the label never occurs in the gold data
    f1: float | None
    support: int              # gold count
    predicted: int


def confusion_matrix(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    """Rows = gold label, columns = predicted label. Unknown predictions are ignored."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred, strict=True):
        if g in index and p in index:
            matrix[index[g]][index[p]] += 1
    return matrix


def per_label(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> list[LabelMetrics]:
    out = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred, strict=True))
        n_pred = sum(p == label for p in pred)
        n_gold = sum(g == label for g in gold)
        precision = tp / n_pred if n_pred else None
        recall = tp / n_gold if n_gold else None
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out.append(LabelMetrics(label, precision, recall, f1, n_gold, n_pred))
    return out


def macro(values: Sequence[float | None]) -> float | None:
    """Mean of defined values (undefined labels are excluded and should be reported)."""
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    if not gold:
        return None
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    if not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[str]], categories: Sequence[str]) -> float | None:
    """``ratings[i]`` = the labels all raters gave item i (same number of raters per item)."""
    if not ratings:
        return None
    n = len(ratings[0])
    if n < 2 or any(len(r) != n for r in ratings):
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of ratings per item")
    N = len(ratings)
    counts = [[r.count(c) for c in categories] for r in ratings]
    p_j = [sum(row[j] for row in counts) / (N * n) for j in range(len(categories))]
    P_i = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    return 1.0 if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def majority_label(labels: Sequence[str]) -> str | None:
    """Strict majority (> half of the given labels), else None (unresolved disagreement)."""
    labels = [label for label in labels if label]
    if not labels:
        return None
    label, count = Counter(labels).most_common(1)[0]
    return label if count * 2 > len(labels) else None
