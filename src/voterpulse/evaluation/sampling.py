"""Export a stratified sample of collected headlines into the annotation template.

The exported file has empty gold-label columns: people fill them in. Nothing in this
project ever fills gold labels automatically.
"""

from __future__ import annotations

import csv
import random
import sqlite3
from pathlib import Path

from voterpulse.config import AppConfig

TEMPLATE_COLUMNS = [
    "item_id", "headline", "source", "url", "published_at",
    # Issue annotation (one primary label from config/issues.yaml, optional secondary)
    "gold_issue", "gold_secondary", "issue_annotator", "issue_notes",
    # Sentiment annotation by three independent annotators: Positive / Neutral / Negative
    "sentiment_annotator_1", "sentiment_annotator_2", "sentiment_annotator_3", "sentiment_notes",
]


def export_sample(conn: sqlite3.Connection, cfg: AppConfig, out_path: Path, n: int = 300,
                  seed: int = 2026) -> int:
    """Write up to ``n`` headlines, spread evenly across RSS feeds, in random order."""
    excluded = {ex.headline.casefold() for ex in cfg.issues.prompt_examples}
    rows = conn.execute(
        "SELECT id, headline, source, url, published_at, feed_key FROM articles WHERE is_synthetic = 0"
    ).fetchall()
    rows = [r for r in rows if r["headline"].casefold() not in excluded]
    rng = random.Random(seed)
    by_feed: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_feed.setdefault(row["feed_key"] or "unknown", []).append(row)
    for items in by_feed.values():
        rng.shuffle(items)
    picked: list[sqlite3.Row] = []
    feeds = sorted(by_feed)
    while len(picked) < n and any(by_feed.values()):
        for feed in feeds:
            if by_feed[feed] and len(picked) < n:
                picked.append(by_feed[feed].pop())
    rng.shuffle(picked)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(picked, start=1):
            writer.writerow({"item_id": f"H{i:03d}", "headline": row["headline"], "source": row["source"] or "",
                             "url": row["url"] or "", "published_at": row["published_at"] or ""})
    return len(picked)


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"item_id", "headline"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]
