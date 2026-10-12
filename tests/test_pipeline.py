import requests

from conftest import NOW, FakeSession, ollama_answer, ollama_tags
from voterpulse.classification.llm_classifier import OllamaClassifier
from voterpulse.models import FeedResult, RawHeadline
from voterpulse.monitoring.pipeline import reclassify_fallbacks, run_pipeline


def _feeds():
    def h(text, key="jobs"):
        return RawHeadline(text, f"https://x.test/{abs(hash(text))}", "Src", NOW, key)
    return [
        FeedResult("jobs", True, [h("Youth unemployment dominates campaign rallies in Bihar"),
                                  h("Power struggle in state cabinet ahead of polls")]),
        FeedResult("farmers", True, [h("MSP hike fails to calm mandi protests", "farmers"),
                                     h("Youth unemployment dominates campaign rallies in Bihar", "farmers")]),
        FeedResult("prices", False, error="HTTP 503"),
    ]


ANSWERS = {
    "Youth unemployment": ("Jobs and Unemployment", "Youth unemployment"),
    "Power struggle": ("Not an election issue", "Power struggle"),
    "MSP hike": ("Farmers and Agriculture", "MSP hike ... mandi protests"),
}


def _llm(cfg):
    def post(url, payload, **kw):
        text = payload["messages"][-1]["content"]
        for key, (issue, ev) in ANSWERS.items():
            if key in text:
                return ollama_answer(issue, ev)
        raise AssertionError(text)
    session = FakeSession(get=lambda url, **kw: ollama_tags(), post=post)
    return OllamaClassifier.from_config(cfg, session=session, sleep=lambda s: None), session


def test_end_to_end_with_llm(cfg, conn):
    llm, session = _llm(cfg)
    report = run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=llm, now=NOW)
    s = report.stats
    assert report.status == "partial"                     # one feed failed
    assert (s.feeds_ok, s.feeds_failed) == (2, 1)
    assert (s.headlines_collected, s.new_headlines, s.duplicates) == (4, 3, 1)
    assert (s.classified, s.llm_classified, s.rejected, s.failed) == (3, 3, 1, 0)
    # rejected headline stored for audit but not scored
    rows = conn.execute("""SELECT a.headline, c.primary_issue, c.method, s.label FROM articles a
                           JOIN classifications c ON c.article_id = a.id
                           LEFT JOIN sentiment_scores s ON s.article_id = a.id""").fetchall()
    by = {r["headline"]: r for r in rows}
    assert by["Power struggle in state cabinet ahead of polls"]["label"] is None
    assert by["MSP hike fails to calm mandi protests"]["label"] is not None
    assert all(r["method"] == "llm" for r in rows)
    run = conn.execute("SELECT * FROM pipeline_runs").fetchone()
    assert run["status"] == "partial" and "prices" in run["error_info"]

    # Second run: everything is a duplicate, no new LLM calls.
    calls = len(session.post_calls)
    report2 = run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=llm, now=NOW)
    assert report2.stats.new_headlines == 0 and report2.stats.duplicates == 4
    assert len(session.post_calls) == calls


def test_end_to_end_ollama_down_uses_fallback(cfg, conn):
    session = FakeSession(get=lambda url, **kw: requests.ConnectionError("refused"))
    llm = OllamaClassifier.from_config(cfg, session=session, sleep=lambda s: None)
    report = run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=llm, now=NOW)
    assert report.stats.fallback_classified == 3 and report.stats.llm_classified == 0
    assert report.llm_note and "circuit breaker" in report.llm_note
    methods = {r[0] for r in conn.execute("SELECT method FROM classifications")}
    assert methods == {"keyword_fallback"}
    assert "keyword fallback used" in conn.execute("SELECT error_info FROM pipeline_runs").fetchone()[0]

    # Once Ollama is back, fallbacks can be re-queued and re-done by the LLM.
    assert reclassify_fallbacks(conn) == 3
    good, _ = _llm(cfg)
    report2 = run_pipeline(cfg, conn, fetcher=lambda rss: [], llm=good, now=NOW)
    assert report2.stats.llm_classified == 3
    assert report2.status == "failed"   # no feeds succeeded (offline), but backlog was processed


def test_offline_run_keeps_existing_data(cfg, conn):
    run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=None, now=NOW)
    before = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    offline = [FeedResult(f.key, False, error="ConnectionError") for f in cfg.rss.feeds]
    report = run_pipeline(cfg, conn, fetcher=lambda rss: offline, llm=None, now=NOW)
    assert report.status == "failed"
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == before


def test_bad_article_does_not_crash_run(cfg, conn, monkeypatch):
    from voterpulse.classification import service

    original = service.HeadlineClassifier.classify

    def flaky(self, headline, headline_hash=None):
        if "MSP" in headline:
            raise RuntimeError("corrupt row")
        return original(self, headline, headline_hash)

    monkeypatch.setattr(service.HeadlineClassifier, "classify", flaky)
    report = run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=None, now=NOW)
    assert report.stats.failed == 1 and report.stats.classified == 2
    assert report.status == "partial"


def test_disagreement_recorded_in_review_queue(cfg, conn):
    def post(url, payload, **kw):
        return ollama_answer("Law and Order", "Youth")   # wrong on purpose
    session = FakeSession(get=lambda url, **kw: ollama_tags(), post=post)
    llm = OllamaClassifier.from_config(cfg, session=session, sleep=lambda s: None)
    feeds = [FeedResult("jobs", True, [RawHeadline("Youth unemployment rises", None, None, None, "jobs")])]
    run_pipeline(cfg, conn, fetcher=lambda rss: feeds, llm=llm, now=NOW)
    row = conn.execute("SELECT * FROM review_queue").fetchone()
    assert row["llm_label"] == "Law and Order" and row["keyword_label"] == "Jobs and Unemployment"
    assert row["reason"] == "llm_keyword_disagreement" and row["status"] == "pending"
    # the valid LLM label is retained
    assert conn.execute("SELECT primary_issue FROM classifications").fetchone()[0] == "Law and Order"


def test_concurrent_run_is_skipped(cfg, conn):
    from voterpulse import database as db
    db.start_run(conn, "other", 30)
    report = run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=None, now=NOW)
    assert report.status == "skipped"


def test_reclassify_includes_invalid_llm_answers(cfg, conn):
    from conftest import ollama_chat
    bad = FakeSession(get=lambda url, **kw: ollama_tags(), post=lambda url, payload, **kw: ollama_chat("nonsense"))
    llm = OllamaClassifier.from_config(cfg, session=bad, sleep=lambda s: None)
    run_pipeline(cfg, conn, fetcher=lambda rss: _feeds(), llm=llm, now=NOW)
    assert {r[0] for r in conn.execute("SELECT validation_status FROM classifications")} == {"llm_invalid"}
    assert reclassify_fallbacks(conn) == 3
    good, _ = _llm(cfg)
    assert run_pipeline(cfg, conn, fetcher=lambda rss: [], llm=good, now=NOW).stats.llm_classified == 3


def test_mongo_outage_retains_collection_and_reports_retry(cfg, conn, monkeypatch):
    from voterpulse.big_data import mongo_store

    cfg.settings.big_data.enabled = True

    def unavailable(*args, **kwargs):
        raise RuntimeError("mongodb://private-user:private-password@server")

    monkeypatch.setattr(mongo_store, "sync_articles", unavailable)
    feeds = [f for f in _feeds() if f.ok]
    report = run_pipeline(cfg, conn, fetcher=lambda rss: feeds, llm=False, now=NOW)
    assert report.status == "partial"
    assert report.stats.new_headlines == 3
    assert report.stats.classified == 3
    assert mongo_store.pending_articles(conn) == 3
    assert "retry" in report.mongo_error
    error = conn.execute("SELECT error_info FROM pipeline_runs").fetchone()[0]
    assert "MongoDB sync failed" in error
    assert "private-password" not in error


def test_enabled_pipeline_syncs_live_after_classification(cfg, conn, monkeypatch):
    from voterpulse.big_data import mongo_store

    cfg.settings.big_data.enabled = True
    calls = []

    def sync(connection, config, kind):
        assert connection.execute("SELECT COUNT(*) FROM classifications").fetchone()[0] == 3
        calls.append(kind)
        return 3

    monkeypatch.setattr(mongo_store, "sync_articles", sync)
    cfg.settings.demo_enabled = True  # collector always targets live, even with this flag
    report = run_pipeline(cfg, conn, fetcher=lambda rss: [f for f in _feeds() if f.ok], llm=False, now=NOW)
    assert calls == ["live"]
    assert report.mongo_synced == 3
    assert report.status == "success"
