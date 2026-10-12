"""Generate a clearly synthetic demo dataset in a SEPARATE database.

    python scripts/generate_demo_data.py            # (re)creates data/demo/voterpulse_demo.db
    VOTERPULSE_DEMO_MODE=1 streamlit run app.py     # view it  (Windows: set VOTERPULSE_DEMO_MODE=1)

Every headline is invented (set in Tamil Nadu places), prefixed with "[DEMO]", attributed to fictional
"(synthetic)" publications and stored with is_synthetic=1 in a database stamped
"demo". The live database is never touched. Classification uses the keyword
classifier (recorded honestly as keyword_fallback — no LLM is involved) and real
VADER scoring, so the dashboard logic runs exactly as it does on live data.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone

import _bootstrap  # noqa: F401

from voterpulse import database as db
from voterpulse.classification.region_filter import RegionFilter
from voterpulse.classification.service import HeadlineClassifier
from voterpulse.config import get_config
from voterpulse.ingestion.deduplicator import prepare
from voterpulse.logging_utils import fields, get_logger, setup_from_config
from voterpulse.models import RawHeadline, to_utc_iso
from voterpulse.monitoring.alerts import evaluate_alerts
from voterpulse.sentiment.vader_analyzer import VaderAnalyzer

log = get_logger("scripts.generate_demo_data")

PLACES = ["Chennai", "Madurai", "Coimbatore", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode",
          "Vellore", "Thanjavur", "Thoothukudi", "Kanchipuram", "Dindigul", "Cuddalore", "Nagapattinam",
          "Tiruvarur", "Kanyakumari", "Krishnagiri", "Namakkal", "Villupuram", "Ramanathapuram"]
SOURCES = ["Demo Herald (synthetic)", "Sample Times (synthetic)", "Example Express (synthetic)",
           "Placeholder Post (synthetic)", "Mock Chronicle (synthetic)"]

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "jobs": {
        "pos": ["{place} recruitment drive fills {n} government jobs, youth welcome move",
                "New factories promise {n} jobs in {place}, boosting hopes",
                "Employment scheme helps {n} families find steady work in {place}"],
        "neg": ["Unemployment crisis deepens in {place} as {n} graduates struggle to find jobs",
                "Layoffs hit {n} workers in {place}, anger grows ahead of polls",
                "Jobless youth protest delayed recruitment of {n} posts in {place}"],
        "neu": ["Parties outline jobs plans in {place} manifestos ({n} pages)",
                "{place} releases employment data for ward {n}",
                "Candidates debate MGNREGA work days in {place} constituency {n}"],
    },
    "prices": {
        "pos": ["LPG cylinder price cut brings relief to households in {place} district {n}",
                "Vegetable prices ease in {n} {place} markets, shoppers relieved"],
        "neg": ["Soaring inflation hurts poor families in {place} ward {n}, opposition attacks government",
                "Onion price rise angers households in {place} town {n}",
                "Petrol price hike burdens commuters in {place} city {n}"],
        "neu": ["Price rise discussed at {place} campaign stop {n}",
                "{place} retail inflation figures released for district {n}"],
    },
    "law_order": {
        "pos": ["{place} police praised as crime rate falls in {n} districts, residents feel safer",
                "New women's safety helpline wins support in {place} district {n}"],
        "neg": ["Violence at rally leaves {n} injured in {place}",
                "Rising crime against women sparks outrage in {place} district {n}"],
        "neu": ["Police deploy {n} extra companies for polling in {place}",
                "{place} parties discuss law and order in debate {n}"],
    },
    "farmers": {
        "pos": ["MSP increase welcomed by farmers in {place} district {n}",
                "Record crop procurement brings relief to {n} {place} farmers"],
        "neg": ["Farmers protest delayed MSP payments in {place} district {n}",
                "Crop failure leaves {n} {place} farmers in debt and despair"],
        "neu": ["Parties pitch farm loan waiver in {place} manifestos ({n} pages)",
                "{place} mandi arrivals reported for week {n}"],
    },
    "utilities": {
        "pos": ["{place} promises 24-hour power supply to {n} villages, residents hopeful",
                "Free electricity units scheme praised in {place} ward {n}"],
        "neg": ["Power cuts anger residents of {n} {place} villages as summer peaks",
                "Drinking water shortage hits {n} villages in {place}"],
        "neu": ["{place} discoms announce power tariff review for zone {n}"],
    },
    "health_education": {
        "pos": ["New hospital opens in {place}, improving access to healthcare for {n} villages",
                "Free laptops for students win praise in {place} district {n}"],
        "neg": ["Shortage of doctors hurts hospitals in {place} district {n}",
                "School dropout rise worries parents in {n} {place} villages"],
        "neu": ["Parties promise schools and hospitals in {place} manifestos ({n} pages)",
                "{place} education budget presented for district {n}",
                "{place} health department publishes hospital list for zone {n}"],
    },
    "irrelevant": {
        "neu": ["Cricket team wins series at {place} stadium in match {n}",
                "Film star visits temple in {place} town {n}",
                "Candidates file nomination papers in {place} seat {n}",
                "Power struggle in {place} cabinet over portfolio {n}"],
    },
}

# (headlines per day, {phase: (p_pos, p_neg)}) — phases: "old" (8-29 days ago),
# "week" (1-7 days ago), "today" (last 24 hours). Remaining probability is neutral.
SCENARIO = {
    "jobs": (8, {"old": (0.30, 0.35), "week": (0.20, 0.50), "today": (0.05, 0.85)}),
    "prices": (6, {"old": (0.10, 0.65), "week": (0.10, 0.65), "today": (0.10, 0.65)}),
    "law_order": (5, {"old": (0.35, 0.35), "week": (0.35, 0.35), "today": (0.35, 0.35)}),
    "farmers": (6, {"old": (0.20, 0.50), "week": (0.50, 0.20), "today": (0.50, 0.20)}),
    "utilities": (1.4, {"old": (0.30, 0.40), "week": (0.30, 0.40), "today": (0.30, 0.40)}),
    "health_education": (5, {"old": (0.20, 0.12), "week": (0.20, 0.12), "today": (0.20, 0.12)}),
    "irrelevant": (3, {"old": (0, 0), "week": (0, 0), "today": (0, 0)}),
}
EXTRA_TODAY = {"jobs": 40}  # concentrated burst that should trigger a mood-shift alert


def _pick(rng: random.Random, key: str, phase: str) -> str:
    p_pos, p_neg = SCENARIO[key][1][phase]
    roll = rng.random()
    tone = "pos" if roll < p_pos else "neg" if roll < p_pos + p_neg else "neu"
    options = TEMPLATES[key].get(tone) or TEMPLATES[key]["neu"]
    return rng.choice(options).format(place=rng.choice(PLACES), n=rng.randint(2, 999))


def generate(days: int = 30, seed: int = 7) -> dict[str, int]:
    cfg = get_config()
    path = cfg.demo_db_path
    if path.resolve() == cfg.live_db_path.resolve():
        raise SystemExit("Refusing: demo_path and live database path are the same file.")
    conn = db.init_db(path, "demo")
    # Keep the journal identity and Mongo deletion queue when regenerating. Unlinking
    # this file would leave the previous demo corpus in MongoDB without tombstones.
    with db.transaction(conn):
        conn.execute("DELETE FROM articles")
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM pipeline_runs")
        conn.execute("DELETE FROM classification_cache")
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    classifier = HeadlineClassifier(cfg.issues, llm=None)
    region_filter = RegionFilter(cfg.settings.region)
    vader = VaderAnalyzer(cfg.thresholds.headline)
    stats = {"inserted": 0, "duplicates": 0, "accepted": 0, "rejected": 0, "issue_mismatch": 0}
    expected_name = {i.key: i.name for i in cfg.issues.issues}

    run_id = db.start_run(conn, "demo_generator", 30)
    with db.transaction(conn):
        for key, (per_day, _phases) in SCENARIO.items():
            total = int(per_day * days)
            jobs = [(rng.uniform(0, days * 24), None) for _ in range(total)]
            jobs += [(rng.uniform(0, 23.5), "today") for _ in range(EXTRA_TODAY.get(key, 0))]
            for hours_ago, forced in jobs:
                phase = forced or ("today" if hours_ago < 24 else "week" if hours_ago < 8 * 24 else "old")
                text = "[DEMO] " + _pick(rng, key, phase)
                published = now - timedelta(hours=hours_ago)
                ingested = min(now, published + timedelta(minutes=rng.randint(3, 40)))
                article = prepare(RawHeadline(text, f"https://example.invalid/demo/{rng.getrandbits(48):x}",
                                              rng.choice(SOURCES), published, key))
                article_id = db.insert_article(conn, article, ingested_at=ingested, is_synthetic=True)
                if article_id is None:
                    stats["duplicates"] += 1
                    continue
                stats["inserted"] += 1
                outcome = classifier.classify(text)
                outcome.error = "synthetic demo data: keyword classifier used, no LLM involved"
                relevant = outcome.primary_issue != cfg.issues.not_relevant_label
                region = region_filter.check(text)
                db.save_classification(conn, article_id, outcome, relevant, ingested,
                                       region_status=region.status, region_evidence=region.evidence)
                if relevant:
                    db.save_sentiment(conn, article_id, vader.score(text), ingested)
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1
                if outcome.primary_issue != expected_name.get(key, cfg.issues.not_relevant_label):
                    stats["issue_mismatch"] += 1
    if run_id is not None:
        db.finish_run(conn, run_id, "success", {
            "headlines_collected": stats["inserted"] + stats["duplicates"], "new_headlines": stats["inserted"],
            "duplicates": stats["duplicates"], "classified": stats["inserted"],
            "fallback_classified": stats["inserted"], "rejected": stats["rejected"], "feeds_ok": len(SCENARIO),
            "error_info": "synthetic demo data generated locally; no RSS or LLM used"})
    evaluate_alerts(conn, cfg, now)
    conn.close()
    log.info("demo_generated", extra=fields(path=path, **stats))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    setup_from_config()
    stats = generate(args.days, args.seed)
    print(f"Demo database written to {get_config().demo_db_path}")
    print(stats)
    print("Start the dashboard in demo mode:  VOTERPULSE_DEMO_MODE=1 streamlit run app.py")


if __name__ == "__main__":
    main()
