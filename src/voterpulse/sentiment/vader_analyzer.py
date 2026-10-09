"""VADER headline sentiment, independent of issue classification.

VADER is a lexicon/rule-based scorer tuned on English social-media text. It is not
trained on Indian political reporting, it reads only the headline, and it cannot
detect sarcasm reliably. Its scores describe the wording of a headline — not the
opinion of voters.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from voterpulse.config import HeadlineThresholds
from voterpulse.models import SentimentLabel, SentimentScores


def label_for(compound: float, thresholds: HeadlineThresholds) -> SentimentLabel:
    """Map a compound score to a label. Boundaries are inclusive (>= positive_min)."""
    if compound >= thresholds.positive_min:
        return SentimentLabel.POSITIVE
    if compound <= thresholds.negative_max:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


class VaderAnalyzer:
    def __init__(self, thresholds: HeadlineThresholds) -> None:
        self.thresholds = thresholds
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> SentimentScores:
        raw = self._analyzer.polarity_scores(text or "")
        compound = max(-1.0, min(1.0, float(raw["compound"])))
        return SentimentScores(
            positive=float(raw["pos"]),
            negative=float(raw["neg"]),
            neutral=float(raw["neu"]),
            compound=compound,
            label=label_for(compound, self.thresholds),
        )
