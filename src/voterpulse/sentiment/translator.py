"""Plain-language translation engine.

Turns aggregate numbers into short, cautious statements a citizen can read in a few
seconds. Every rule is a pure function driven by ``config/sentiment_thresholds.yaml``;
rationale and boundary examples live in ``docs/thresholds.md``.

Language rules (deliberate):

* Always talk about *coverage* / *headlines*, never "voters", "people" or "approval".
* Always show the sample size next to a conclusion.
* Below the minimum sample size, say there is not enough data — never guess.
* No exclamation marks, no alarm words.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from voterpulse.config import AggregateThresholds, TrendThresholds
from voterpulse.sentiment.aggregator import IssueSummary, TrendResult, Window


class Mood(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient"


class Trend(str, Enum):
    MORE_POSITIVE = "more_positive"
    MORE_NEGATIVE = "more_negative"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


# Colour is never used alone: every colour is paired with MOOD_LABELS text and an icon.
MOOD_COLORS = {
    Mood.POSITIVE: "#1E7B46",     # green
    Mood.NEGATIVE: "#B42318",     # red
    Mood.MIXED: "#B54708",        # amber
    Mood.NEUTRAL: "#B54708",      # amber: no clear direction
    Mood.INSUFFICIENT: "#667085",  # grey
}
MOOD_BACKGROUNDS = {
    Mood.POSITIVE: "#ECFDF3",
    Mood.NEGATIVE: "#FEF3F2",
    Mood.MIXED: "#FFFAEB",
    Mood.NEUTRAL: "#FFFAEB",
    Mood.INSUFFICIENT: "#F2F4F7",
}
MOOD_LABELS = {
    Mood.POSITIVE: "Mostly positive",
    Mood.NEGATIVE: "Mostly negative",
    Mood.MIXED: "Mixed",
    Mood.NEUTRAL: "Mostly neutral",
    Mood.INSUFFICIENT: "Not enough data",
}
MOOD_ICONS = {
    Mood.POSITIVE: "▲",
    Mood.NEGATIVE: "▼",
    Mood.MIXED: "◆",
    Mood.NEUTRAL: "●",
    Mood.INSUFFICIENT: "○",
}
TREND_LABELS = {
    Trend.MORE_POSITIVE: "↗ Becoming more positive",
    Trend.MORE_NEGATIVE: "↘ Becoming more negative",
    Trend.UNCHANGED: "→ Broadly unchanged",
    Trend.UNKNOWN: "– Trend not available",
}


# --------------------------------------------------------------------------- rules
def classify_mood(count: int, mean: float | None, positive_share: float, negative_share: float,
                  neutral_share: float, t: AggregateThresholds, min_headlines: int) -> Mood:
    """Score-to-mood rule. Order matters: sample size first, then neutrality, then direction."""
    if count < min_headlines or mean is None:
        return Mood.INSUFFICIENT
    if neutral_share >= t.neutral_dominant_share and t.negative_mean < mean < t.positive_mean:
        return Mood.NEUTRAL
    if mean >= t.positive_mean and positive_share - negative_share >= t.share_margin:
        return Mood.POSITIVE
    if mean <= t.negative_mean and negative_share - positive_share >= t.share_margin:
        return Mood.NEGATIVE
    return Mood.MIXED


def classify_trend(delta: float | None, t: TrendThresholds) -> Trend:
    if delta is None:
        return Trend.UNKNOWN
    if delta >= t.change_min:
        return Trend.MORE_POSITIVE
    if delta <= -t.change_min:
        return Trend.MORE_NEGATIVE
    return Trend.UNCHANGED


_HEADLINES: dict[tuple[Mood, Trend], str] = {
    (Mood.POSITIVE, Trend.MORE_POSITIVE): "Mostly positive, and improving.",
    (Mood.POSITIVE, Trend.MORE_NEGATIVE): "Mostly positive, but less so than the week before.",
    (Mood.POSITIVE, Trend.UNCHANGED): "Mostly positive, and broadly steady.",
    (Mood.NEGATIVE, Trend.MORE_NEGATIVE): "Mostly negative, and getting worse.",
    (Mood.NEGATIVE, Trend.MORE_POSITIVE): "Mostly negative, but improving.",
    (Mood.NEGATIVE, Trend.UNCHANGED): "Mostly negative, and broadly steady.",
    (Mood.MIXED, Trend.MORE_POSITIVE): "Mixed, and turning more positive.",
    (Mood.MIXED, Trend.MORE_NEGATIVE): "Mixed, and turning more negative.",
    (Mood.MIXED, Trend.UNCHANGED): "Mixed, with no clear direction.",
    (Mood.NEUTRAL, Trend.MORE_POSITIVE): "Mostly neutral, with a slight turn towards positive.",
    (Mood.NEUTRAL, Trend.MORE_NEGATIVE): "Mostly neutral, with a slight turn towards negative.",
    (Mood.NEUTRAL, Trend.UNCHANGED): "Mostly neutral, and broadly steady.",
}
_HEADLINE_NO_TREND = {
    Mood.POSITIVE: "Mostly positive.",
    Mood.NEGATIVE: "Mostly negative.",
    Mood.MIXED: "Mixed, with no clear direction.",
    Mood.NEUTRAL: "Mostly neutral.",
    Mood.INSUFFICIENT: "Not enough data.",
}


def mood_headline(mood: Mood, trend: Trend) -> str:
    """The one-line "Mood:" statement shown on a card."""
    if mood is Mood.INSUFFICIENT or trend is Trend.UNKNOWN:
        return _HEADLINE_NO_TREND[mood]
    return _HEADLINES[(mood, trend)]


def mood_sentence(mood: Mood) -> str:
    return {
        Mood.POSITIVE: "Coverage is mostly positive.",
        Mood.NEGATIVE: "Coverage is mostly negative.",
        Mood.MIXED: "Coverage is mixed, with no clear direction.",
        Mood.NEUTRAL: "Most headlines are neutral, factual reporting with little emotional tone.",
        Mood.INSUFFICIENT: "There is not enough news coverage to draw a reliable conclusion.",
    }[mood]


def trend_sentence(trend: Trend, trend_result: TrendResult | None = None) -> str:
    if trend is Trend.MORE_NEGATIVE:
        return "Negative coverage has increased compared with the previous week."
    if trend is Trend.MORE_POSITIVE:
        return "Coverage is becoming more positive compared with the previous week."
    if trend is Trend.UNCHANGED:
        return "Coverage remains broadly unchanged compared with the previous week."
    if trend_result is not None and trend_result.previous.is_partial:
        return "Collection started too recently to compare with the previous week."
    return "There is not enough coverage in both weeks to say how the tone is changing."


def _pct(share: float) -> str:
    return f"{round(share * 100)}%"


def distribution_sentence(summary: IssueSummary) -> str | None:
    if summary.count == 0:
        return None
    noun = "headline" if summary.count == 1 else "headlines"
    return (f"Of {summary.count} {noun}, {_pct(summary.negative_share)} read as negative, "
            f"{_pct(summary.positive_share)} as positive and {_pct(summary.neutral_share)} as neutral.")


def period_phrase(window: Window) -> str:
    return {
        "24h": "over the last 24 hours",
        "7d": "over the last seven days",
        "30d": "over the last 30 days",
    }.get(window.key, f"between {window.label}")


def sample_sentence(count: int, window: Window, min_headlines: int) -> str:
    noun = "news headline" if count == 1 else "news headlines"
    base = f"Based on {count} {noun} collected {period_phrase(window)}."
    if count < min_headlines:
        base += f" At least {min_headlines} are needed before VoterPulse describes a mood."
    if window.is_partial:
        base += " Collection began part-way through this period, so it is only partly covered."
    return base


def humanize_age(age: timedelta) -> str:
    seconds = max(0, int(age.total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"{days} days ago"


def freshness_sentence(last_update: datetime | None, now: datetime, stale_after_minutes: int) -> str:
    if last_update is None:
        return "No successful update yet."
    age = now - last_update
    if age > timedelta(minutes=stale_after_minutes):
        return f"Data may be out of date: last updated {humanize_age(age)}."
    return f"Updated {humanize_age(age)}."


# --------------------------------------------------------------------------- report
@dataclass
class MoodReport:
    issue: str
    mood: Mood
    trend: Trend
    headline: str            # "Mostly negative, and getting worse."
    explanation: list[str]   # plain sentences, most important first
    sample: str
    freshness: str
    count: int

    @property
    def color(self) -> str:
        return MOOD_COLORS[self.mood]

    @property
    def background(self) -> str:
        return MOOD_BACKGROUNDS[self.mood]

    @property
    def label(self) -> str:
        return MOOD_LABELS[self.mood]

    @property
    def icon(self) -> str:
        return MOOD_ICONS[self.mood]

    @property
    def trend_label(self) -> str:
        return TREND_LABELS[self.trend]


def build_report(summary: IssueSummary, trend_result: TrendResult | None,
                 aggregate_t: AggregateThresholds, trend_t: TrendThresholds, min_headlines: int,
                 last_update: datetime | None, now: datetime, stale_after_minutes: int) -> MoodReport:
    mood = classify_mood(summary.count, summary.mean_compound, summary.positive_share,
                         summary.negative_share, summary.neutral_share, aggregate_t, min_headlines)
    trend = classify_trend(trend_result.delta if trend_result else None, trend_t)
    explanation = [mood_sentence(mood)]
    if mood is not Mood.INSUFFICIENT:
        dist = distribution_sentence(summary)
        if dist:
            explanation.append(dist)
    explanation.append(trend_sentence(trend, trend_result))
    return MoodReport(
        issue=summary.issue,
        mood=mood,
        trend=trend,
        headline=mood_headline(mood, trend),
        explanation=explanation,
        sample=sample_sentence(summary.count, summary.window, min_headlines),
        freshness=freshness_sentence(last_update, now, stale_after_minutes),
        count=summary.count,
    )
