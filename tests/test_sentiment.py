from datetime import date, timedelta

import pytest

from conftest import NOW, seed
from voterpulse.config import HeadlineThresholds
from voterpulse.models import SentimentLabel
from voterpulse.sentiment import aggregator as agg
from voterpulse.sentiment.vader_analyzer import VaderAnalyzer, label_for

JOBS = "Jobs and Unemployment"


@pytest.mark.parametrize("compound,label", [
    (0.05, SentimentLabel.POSITIVE), (0.0499, SentimentLabel.NEUTRAL), (0.0, SentimentLabel.NEUTRAL),
    (-0.0499, SentimentLabel.NEUTRAL), (-0.05, SentimentLabel.NEGATIVE), (1.0, SentimentLabel.POSITIVE),
    (-1.0, SentimentLabel.NEGATIVE),
])
def test_headline_label_boundaries(compound, label):
    assert label_for(compound, HeadlineThresholds()) is label


def test_vader_scores_are_complete_and_deterministic():
    v = VaderAnalyzer(HeadlineThresholds())
    a = v.score("Farmers angry as crop prices crash")
    b = v.score("Farmers angry as crop prices crash")
    assert a == b
    assert a.label is SentimentLabel.NEGATIVE and a.compound < 0
    assert abs(a.positive + a.negative + a.neutral - 1.0) < 0.01
    assert v.score("Scheme brings relief and new hope to farmers").label is SentimentLabel.POSITIVE
    assert v.score("").label is SentimentLabel.NEUTRAL


def test_summary_counts_and_excludes_rejected(conn):
    seed(conn, JOBS, -0.5, NOW - timedelta(hours=2), n=3)
    seed(conn, JOBS, 0.6, NOW - timedelta(hours=3), n=1)
    seed(conn, JOBS, 0.0, NOW - timedelta(hours=3), n=1)
    seed(conn, JOBS, 0.9, NOW - timedelta(hours=1), n=5, relevant=False)   # rejected
    df = agg.load_relevant_frame(conn)
    s = agg.summarize(df, JOBS, agg.rolling_window("24h", NOW), min_headlines=25)
    assert s.count == 5
    assert (s.positive_count, s.negative_count, s.neutral_count) == (1, 3, 1)
    assert s.mean_compound == pytest.approx((-1.5 + 0.6) / 5)
    assert s.sufficient is False
    assert s.negative_share == pytest.approx(0.6)


def test_window_boundaries_are_half_open(conn):
    seed(conn, JOBS, -0.5, NOW - timedelta(hours=24))            # exactly at start -> included
    seed(conn, JOBS, -0.5, NOW - timedelta(hours=24, seconds=1))  # just before -> excluded
    seed(conn, JOBS, -0.5, NOW)                                    # at end -> excluded
    df = agg.load_relevant_frame(conn)
    assert agg.summarize(df, JOBS, agg.rolling_window("24h", NOW), 25).count == 1


def test_partial_window_flag(conn):
    w = agg.rolling_window("30d", NOW, data_start=NOW - timedelta(days=3))
    assert w.is_partial
    assert not agg.rolling_window("24h", NOW, data_start=NOW - timedelta(days=3)).is_partial


def test_completed_week_windows_end_at_ist_midnight():
    cur, prev = agg.completed_week_windows(NOW)  # NOW = 12:00 IST on 9 Oct
    assert cur.end.isoformat() == "2026-10-08T18:30:00+00:00"  # 9 Oct 00:00 IST
    assert cur.end - cur.start == timedelta(days=7)
    assert prev.end == cur.start


def test_weekly_trend(conn):
    cur, prev = agg.completed_week_windows(NOW)
    seed(conn, JOBS, -0.6, cur.start + timedelta(hours=5), n=25)
    seed(conn, JOBS, -0.1, prev.start + timedelta(hours=5), n=25)
    seed(conn, JOBS, 0.9, NOW - timedelta(hours=1), n=10)   # today: excluded from trend
    df = agg.load_relevant_frame(conn)
    t = agg.weekly_trend(df, JOBS, NOW, min_each=25)
    assert t.sufficient and t.delta == pytest.approx(-0.5)
    t2 = agg.weekly_trend(df, JOBS, NOW, min_each=26)
    assert not t2.sufficient and t2.delta is None


def test_custom_window_uses_ist_dates():
    w = agg.custom_window(date(2026, 10, 1), date(2026, 10, 7), NOW)
    assert w.start.isoformat() == "2026-09-30T18:30:00+00:00"
    assert w.end.isoformat() == "2026-10-07T18:30:00+00:00"
    with pytest.raises(ValueError):
        agg.custom_window(date(2026, 10, 7), date(2026, 10, 1), NOW)


def test_daily_series_and_sources(conn):
    seed(conn, JOBS, -0.5, NOW - timedelta(days=1), n=2, source="A")
    seed(conn, JOBS, 0.5, NOW - timedelta(days=2), n=1, source="B")
    df = agg.load_relevant_frame(conn)
    series = agg.daily_series(df, JOBS, agg.rolling_window("7d", NOW))
    assert series["headlines"].sum() == 3 and len(series) == 2
    s = agg.summarize(df, JOBS, agg.rolling_window("7d", NOW), 25)
    assert s.sources == [("A", 2), ("B", 1)]


def test_empty_database_aggregates(conn):
    df = agg.load_relevant_frame(conn)
    s = agg.summarize(df, JOBS, agg.rolling_window("7d", NOW), 25)
    assert s.count == 0 and s.mean_compound is None and not s.sufficient
    assert agg.daily_series(df, JOBS, agg.rolling_window("7d", NOW)).empty
