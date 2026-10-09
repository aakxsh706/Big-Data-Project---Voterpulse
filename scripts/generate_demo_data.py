"""Generate a clearly synthetic demo dataset in a SEPARATE database.

    python scripts/generate_demo_data.py            # (re)creates data/demo/voterpulse_demo.db
    VOTERPULSE_DEMO_MODE=1 streamlit run app.py     # view it  (Windows: set VOTERPULSE_DEMO_MODE=1)

Every headline is invented, prefixed with "[DEMO]", attributed to fictional
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
from voterpulse.classification.service import HeadlineClassifier
from voterpulse.config import get_config
from voterpulse.ingestion.deduplicator import prepare
from voterpulse.logging_utils import fields, get_logger, setup_from_config
from voterpulse.models import RawHeadline, to_utc_iso
from voterpulse.monitoring.alerts import evaluate_alerts
from voterpulse.sentiment.vader_analyzer import VaderAnalyzer

log = get_logger("scripts.generate_demo_data")

STATES = ["Bihar", "Punjab", "Kerala", "Assam", "Odisha", "Gujarat", "Haryana", "Jharkhand",
          "Telangana", "Karnataka", "Maharashtra", "Rajasthan", "Tamil Nadu", "West Bengal",
          "Uttar Pradesh", "Madhya Pradesh", "Chhattisgarh", "Goa", "Tripura", "Delhi"]
SOURCES = ["Demo Herald (synthetic)", "Sample Times (synthetic)", "Example Express (synthetic)",
           "Placeholder Post (synthetic)", "Mock Chronicle (synthetic)"]

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "jobs": {
        "pos": ["{state} recruitment drive fills {n} government jobs, youth welcome move",
                "New factories promise {n} jobs in {state}, boosting hopes",
                "Employment scheme helps {n} families find steady work in {state}"],
        "neg": ["Unemployment crisis deepens in {state} as {n} graduates struggle to find jobs",
                "Layoffs hit {n} workers in {state}, anger grows ahead of polls",
                "Jobless youth protest delayed recruitment of {n} posts in {state}"],
        "neu": ["Parties outline jobs plans in {state} manifestos ({n} pages)",
                "{state} releases employment data for ward {n}",
                "Candidates debate MGNREGA work days in {state} constituency {n}"],
    },
    "prices": {
        "pos": ["LPG cylinder price cut brings relief to households in {state} district {n}",
                "Vegetable prices ease in {n} {state} markets, shoppers relieved"],
        "neg": ["Soaring inflation hurts poor families in {state} ward {n}, opposition attacks government",
                "Onion price rise angers households in {state} town {n}",
                "Petrol price hike burdens commuters in {state} city {n}"],
        "neu": ["Price rise discussed at {state} campaign stop {n}",
                "{state} retail inflation figures released for district {n}"],
    },
    "law_order": {
        "pos": ["{state} police praised as crime rate falls in {n} districts, residents feel safer",
                "New women's safety helpline wins support in {state} district {n}"],
        "neg": ["Violence at rally leaves {n} injured in {state}",
                "Rising crime against women sparks outrage in {state} district {n}"],
        "neu": ["Police deploy {n} extra companies for polling in {state}",
                "{state} parties discuss law and order in debate {n}"],
    },
    "farmers": {
        "pos": ["MSP increase welcomed by farmers in {state} district {n}",
                "Record crop procurement brings relief to {n} {state} farmers"],
        "neg": ["Farmers protest delayed MSP payments in {state} district {n}",
                "Crop failure leaves {n} {state} farmers in debt and despair"],
        "neu": ["Parties pitch farm loan waiver in {state} manifestos ({n} pages)",
                "{state} mandi arrivals reported for week {n}"],
    },
    "utilities": {
        "pos": ["{state} promises 24-hour power supply to {n} villages, residents hopeful",
                "Free electricity units scheme praised in {state} town {n}"],
        "neg": ["Power cuts anger residents of {n} {state} villages as summer peaks",
                "Drinking water shortage hits {n} villages in {state}"],
        "neu": ["{state} discoms announce power tariff review for zone {n}"],
    },
    "health_education": {
        "pos": ["New hospital opens in {state}, improving access to healthcare for {n} villages",
                "Free laptops for students win praise in {state} district {n}"],
        "neg": ["Shortage of doctors hurts hospitals in {state} district {n}",
                "School dropout rise worries parents in {n} {state} villages"],
        "neu": ["Parties promise schools and hospitals in {state} manifestos ({n} pages)",
                "{state} education budget presented for district {n}",
                "{state} health department publishes hospital list for zone {n}"],
    },
    "irrelevant": {
        "neu": ["Cricket team wins series at {state} stadium in match {n}",
                "Film star visits temple in {state} town {n}",
                "Candidates file nomination papers in {state} seat {n}",
                "Power struggle in {state} cabinet over portfolio {n}"],
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
    return rng.choice(options).format(state=rng.choice(STATES), n=rng.randint(2, 999))


def generate(days: int = 30, seed: int = 7) -> dict[str, int]:
    cfg = get_config()
    path = cfg.demo_db_path
    if path.resolve() == cfg.live_db_path.resolve():
        raise SystemExit("Refusing: demo_path and live database path are the same file.")
    for suffix in ("", "-wal", "-shm"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists():
            candidate.unlink()
    conn = db.init_db(path, "demo")
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    classifier = HeadlineClassifier(cfg.issues, llm=None)
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
                db.save_classification(conn, article_id, outcome, relevant, ingested)
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
