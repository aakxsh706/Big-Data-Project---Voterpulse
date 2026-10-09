from datetime import datetime, timezone

import requests

from conftest import FakeResponse, FakeSession
from voterpulse import database as db
from voterpulse.ingestion.deduplicator import deduplicate_batch, filter_new
from voterpulse.ingestion.normalizer import clean_headline, headline_hash, normalize_for_dedup
from voterpulse.ingestion.rss_fetcher import build_feed_url, fetch_all, fetch_feed, parse_feed
from voterpulse.models import RawHeadline


def test_clean_headline_strips_source_suffix_and_html():
    assert clean_headline("Jobs &amp; wages <b>rise</b> - The Hindu", "The Hindu") == "Jobs & wages rise"
    assert clean_headline("  Spaces\n  everywhere  ") == "Spaces everywhere"


def test_normalisation_makes_variants_hash_equal():
    a = normalize_for_dedup("Youth unemployment dominates rallies!")
    b = normalize_for_dedup("YOUTH  Unemployment — dominates rallies")
    assert a == b == "youth unemployment dominates rallies"
    assert headline_hash(a) == headline_hash(b)
    assert len(headline_hash(a)) == 64


def test_hash_is_stable():
    # Fixed expected value guards against accidental changes to normalisation.
    assert headline_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_parse_feed_extracts_fields(rss_bytes):
    items, error = parse_feed(rss_bytes, "jobs")
    assert error is None
    assert len(items) == 3  # empty title skipped
    first = items[0]
    assert first.source == "The Example Times"
    assert first.published_at == datetime(2026, 10, 5, 6, 30, tzinfo=timezone.utc)
    assert items[2].published_at is None  # missing date handled


def test_parse_feed_malformed():
    items, error = parse_feed(b"<<<not xml at all", "jobs")
    assert items == [] and error is not None


def test_build_feed_url_uses_india_english(cfg):
    url = build_feed_url(cfg.rss, cfg.rss.feeds[0])
    assert url.startswith("https://news.google.com/rss/search?q=")
    assert "hl=en-IN" in url and "gl=IN" in url and "ceid=IN%3Aen" in url


def test_dedupe_across_feeds(rss_bytes):
    a, _ = parse_feed(rss_bytes, "jobs")
    b, _ = parse_feed(rss_bytes, "prices")
    unique, dupes, unusable = deduplicate_batch(a + b)
    assert len(unique) == 2          # two variants of the same Bihar headline collapse
    assert dupes == 4
    assert unique[0].feed_key == "jobs"  # first occurrence wins


def test_filter_new_against_database(conn, rss_bytes):
    items, _ = parse_feed(rss_bytes, "jobs")
    unique, _, _ = deduplicate_batch(items)
    db.insert_article(conn, unique[0])
    new, stored = filter_new(conn, unique)
    assert stored == 1 and len(new) == 1


def test_fetch_feed_retries_then_succeeds(cfg, rss_bytes, no_sleep):
    responses = iter([requests.ConnectionError("down"), FakeResponse(503, {}), FakeResponse(200, content=rss_bytes)])
    session = FakeSession(get=lambda url, **kw: next(responses))
    result = fetch_feed(cfg.rss, cfg.rss.feeds[0], session, no_sleep)
    assert result.ok and len(result.headlines) == 3
    assert len(session.get_calls) == 3


def test_fetch_feed_gives_up_after_bounded_retries(cfg, no_sleep):
    session = FakeSession(get=lambda url, **kw: requests.Timeout("slow"))
    result = fetch_feed(cfg.rss, cfg.rss.feeds[0], session, no_sleep)
    assert not result.ok and "Timeout" in result.error
    assert len(session.get_calls) == cfg.rss.request.max_retries + 1


def test_fetch_feed_client_error_not_retried(cfg, no_sleep):
    session = FakeSession(get=lambda url, **kw: FakeResponse(403, {}))
    result = fetch_feed(cfg.rss, cfg.rss.feeds[0], session, no_sleep)
    assert not result.ok and result.status_code == 403 and len(session.get_calls) == 1


def test_one_failing_feed_does_not_stop_others(cfg, rss_bytes, no_sleep):
    def get(url, **kw):
        return FakeResponse(500, {}) if "unemployment" in url else FakeResponse(200, content=rss_bytes)
    results = fetch_all(cfg.rss, FakeSession(get=get), no_sleep)
    assert len(results) == len(cfg.rss.feeds)
    assert sum(not r.ok for r in results) == 1


def test_clean_headline_strips_repeated_source_suffix():
    assert clean_headline("Rahul Gandhi leads march - India Today - India Today", "India Today") == "Rahul Gandhi leads march"
