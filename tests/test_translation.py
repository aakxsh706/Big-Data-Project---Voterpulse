"""Boundary tests for every score-to-language rule (see docs/thresholds.md)."""

from datetime import timedelta

import pytest

from conftest import NOW
from voterpulse.config import AggregateThresholds, TrendThresholds
from voterpulse.sentiment.aggregator import IssueSummary, TrendResult, Window, completed_week_windows, rolling_window
from voterpulse.sentiment.translator import (Mood, Trend, build_report, classify_mood, classify_trend,
                                             freshness_sentence, humanize_age, mood_headline)

A, T = AggregateThresholds(), TrendThresholds()


def mood(count=30, mean=0.0, pos=0.3, neg=0.3, neu=0.4, min_h=25):
    return classify_mood(count, mean, pos, neg, neu, A, min_h)


@pytest.mark.parametrize("count,expected", [(24, Mood.INSUFFICIENT), (25, Mood.POSITIVE), (0, Mood.INSUFFICIENT)])
def test_minimum_sample_boundary(count, expected):
    assert mood(count=count, mean=0.5, pos=0.6, neg=0.1, neu=0.3) is expected


def test_none_mean_is_insufficient():
    assert classify_mood(30, None, 0, 0, 0, A, 25) is Mood.INSUFFICIENT


@pytest.mark.parametrize("mean,pos,neg,expected", [
    (0.10, 0.40, 0.30, Mood.POSITIVE),    # both conditions exactly at threshold
    (0.0999, 0.40, 0.30, Mood.MIXED),     # mean just below
    (0.20, 0.40, 0.3001, Mood.MIXED),     # margin just below
    (-0.10, 0.30, 0.40, Mood.NEGATIVE),
    (-0.0999, 0.30, 0.40, Mood.MIXED),
    (-0.30, 0.35, 0.40, Mood.MIXED),      # negative mean but sides close -> mixed
    (0.0, 0.30, 0.30, Mood.MIXED),
])
def test_direction_boundaries(mean, pos, neg, expected):
    assert mood(mean=mean, pos=pos, neg=neg, neu=1 - pos - neg) is expected


@pytest.mark.parametrize("neu,mean,expected", [
    (0.60, 0.05, Mood.NEUTRAL),
    (0.5999, 0.05, Mood.MIXED),
    (0.70, 0.10, Mood.POSITIVE),   # strong enough mean overrides neutrality (needs margin)
    (0.80, -0.0999, Mood.NEUTRAL),
])
def test_neutral_dominance_boundaries(neu, mean, expected):
    pos = (1 - neu) * (0.75 if mean > 0 else 0.25)
    neg = (1 - neu) - pos
    if expected is Mood.POSITIVE:
        pos, neg = 0.25, 0.05
    assert mood(mean=mean, pos=pos, neg=neg, neu=neu) is expected


@pytest.mark.parametrize("delta,expected", [
    (None, Trend.UNKNOWN), (0.05, Trend.MORE_POSITIVE), (0.0499, Trend.UNCHANGED),
    (-0.0499, Trend.UNCHANGED), (-0.05, Trend.MORE_NEGATIVE), (0.0, Trend.UNCHANGED),
])
def test_trend_boundaries(delta, expected):
    assert classify_trend(delta, T) is expected


def test_spec_example_statement():
    assert mood_headline(Mood.NEGATIVE, Trend.MORE_NEGATIVE) == "Mostly negative, and getting worse."
    assert mood_headline(Mood.MIXED, Trend.UNCHANGED) == "Mixed, with no clear direction."
    assert mood_headline(Mood.INSUFFICIENT, Trend.MORE_NEGATIVE) == "Not enough data."
    for m in Mood:
        for t in Trend:
            text = mood_headline(m, t)
            assert text and "!" not in text
            assert "voter" not in text.lower() and "public" not in text.lower()


@pytest.mark.parametrize("minutes,text", [(0, "just now"), (1, "1 minute ago"), (12, "12 minutes ago"),
                                          (60, "1 hour ago"), (300, "5 hours ago"), (60 * 72, "3 days ago")])
def test_humanize_age(minutes, text):
    assert humanize_age(timedelta(minutes=minutes)) == text


def test_freshness_boundary():
    assert freshness_sentence(None, NOW, 60) == "No successful update yet."
    assert freshness_sentence(NOW - timedelta(minutes=60), NOW, 60) == "Updated 1 hour ago."
    assert freshness_sentence(NOW - timedelta(minutes=61), NOW, 60).startswith("Data may be out of date")


def _summary(count, mean, pos, neg, neu, window=None):
    return IssueSummary("Jobs and Unemployment", window or rolling_window("7d", NOW), count, mean,
                        pos, neg, neu, count >= 25, NOW, [])


def test_full_report_matches_spec_example():
    cur, prev = completed_week_windows(NOW)
    trend = TrendResult("Jobs and Unemployment", cur, prev, 80, 70, -0.35, -0.15, True)
    report = build_report(_summary(86, -0.32, 15, 55, 16), trend, A, T, 25,
                          NOW - timedelta(minutes=12), NOW, 60)
    assert report.headline == "Mostly negative, and getting worse."
    assert report.sample == "Based on 86 news headlines collected over the last seven days."
    assert report.freshness == "Updated 12 minutes ago."
    assert report.explanation[0] == "Coverage is mostly negative."
    assert "Negative coverage has increased" in report.explanation[-1]
    assert report.color == "#B42318" and report.label == "Mostly negative"


def test_report_insufficient_is_grey_and_cautious():
    report = build_report(_summary(10, -0.8, 0, 10, 0), None, A, T, 25, NOW, NOW, 60)
    assert report.mood is Mood.INSUFFICIENT and report.label == "Not enough data"
    assert report.color == "#667085"
    assert "not enough news coverage" in report.explanation[0]
    assert "At least 25 are needed" in report.sample
    assert all("negative" not in s.lower() for s in report.explanation[:1])


def test_partial_window_is_disclosed():
    w = Window(NOW - timedelta(days=7), NOW, "7d", "Last 7 days", True)
    report = build_report(_summary(30, 0.0, 9, 9, 12, w), None, A, T, 25, NOW, NOW, 60)
    assert "only partly covered" in report.sample
