"""Region (Tamil Nadu) scoping: deterministic filter, migration and pipeline behaviour."""

import sqlite3

import pytest

from conftest import NOW
from voterpulse import database as db
from voterpulse.classification.region_filter import IN_REGION, NOT_CHECKED, OUT_OF_REGION, RegionFilter
from voterpulse.config import RegionSettings
from voterpulse.models import FeedResult, RawHeadline
from voterpulse.monitoring.pipeline import run_pipeline


@pytest.fixture
def rf(cfg):
    return RegionFilter(cfg.settings.region)


@pytest.mark.parametrize("headline,status", [
    ("Bihar farmers protest delay in paddy procurement", OUT_OF_REGION),
    ("Karnataka hikes milk price by Rs 4 per litre", OUT_OF_REGION),
    ("Mumbai police arrest gang behind cyber fraud", OUT_OF_REGION),
    ("Tamil Nadu records lowest Cauvery realisation", IN_REGION),
    ("TNPDCL rules out eight-hour power cuts", IN_REGION),
    ("Karnataka must release Cauvery water to Tamil Nadu, says panel", IN_REGION),  # both -> local wins
    ("Delay in appointing agriculture secretary betrays farmers", IN_REGION),         # no place -> kept
    ("Chennai and Bengaluru see rise in tech hiring", IN_REGION),
])
def test_region_rule(rf, headline, status):
    assert rf.check(headline).status == status


def test_region_evidence_names_the_place(rf):
    assert rf.check("Bihar farmers protest").evidence == "Bihar"


def test_short_caps_abbreviation_is_case_sensitive(rf):
    assert rf.check("Kerala floods: what tn means").status == OUT_OF_REGION   # 'tn' lowercase ≠ TN
    assert rf.check("Kerala and TN sign water pact").status == IN_REGION


def test_no_region_configured_means_not_checked():
    assert RegionFilter(RegionSettings()).check("Bihar farmers protest").status == NOT_CHECKED


def test_migration_upgrades_existing_v1_database(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(f"BEGIN;{db.MIGRATIONS[0]}PRAGMA user_version = 1;COMMIT;")
    conn.close()
    upgraded = db.init_db(path)
    assert db.schema_version(upgraded) == len(db.MIGRATIONS)
    assert db.schema_version(upgraded) >= 2
    cols = {r[1] for r in upgraded.execute("PRAGMA table_info(classifications)")}
    assert {"region_status", "region_evidence"} <= cols
    upgraded.close()


def test_pipeline_excludes_out_of_region_from_counts(cfg, conn):
    feeds = [FeedResult("farmers", True, [
        RawHeadline("Bihar farmers protest delay in paddy procurement", None, "S", NOW, "farmers"),
        RawHeadline("Delta farmers in Tamil Nadu protest delay in paddy procurement", None, "S", NOW, "farmers"),
    ])]
    report = run_pipeline(cfg, conn, fetcher=lambda rss: feeds, llm=None, now=NOW)
    assert report.stats.out_of_region == 1 and report.stats.rejected == 0
    rows = conn.execute("""SELECT a.headline, c.is_relevant, c.region_status, c.region_evidence, s.label
                           FROM articles a JOIN classifications c ON c.article_id = a.id
                           LEFT JOIN sentiment_scores s ON s.article_id = a.id ORDER BY a.id""").fetchall()
    bihar, tn = rows
    assert (bihar["is_relevant"], bihar["region_status"], bihar["region_evidence"]) == (0, OUT_OF_REGION, "Bihar")
    assert bihar["label"] is None                     # never scored
    assert tn["is_relevant"] == 1 and tn["region_status"] == IN_REGION and tn["label"] is not None
    assert conn.execute("SELECT out_of_region FROM pipeline_runs").fetchone()[0] == 1


def test_recheck_region_after_config_change(cfg, conn):
    from voterpulse.monitoring.pipeline import recheck_region
    feeds = [FeedResult("farmers", True, [
        RawHeadline("Farmers in Hyderabad protest delay in paddy procurement", None, "S", NOW, "farmers")])]
    run_pipeline(cfg, conn, fetcher=lambda rss: feeds, llm=None, now=NOW)
    assert conn.execute("SELECT is_relevant FROM classifications").fetchone()[0] == 0
    cfg.settings.region.other_places.remove("Hyderabad")      # simulate editing settings.yaml
    assert recheck_region(conn, cfg) == {"now_counted": 1, "now_excluded": 0}
    assert conn.execute("SELECT COUNT(*) FROM sentiment_scores").fetchone()[0] == 1
