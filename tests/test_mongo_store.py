"""Durable Mongo synchronization contracts, without a Mongo server or optional driver."""

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import pytest

from conftest import NOW, seed
from voterpulse import database as db
from voterpulse.big_data import mongo_store as store
from voterpulse.ingestion.deduplicator import prepare
from voterpulse.models import RawHeadline


@dataclass
class Replace:
    filter: dict
    replacement: dict
    upsert: bool = True


@dataclass
class Delete:
    filter: dict


class Collection:
    def __init__(self):
        self.documents = {}
        self.batches = []
        self.indexes = []
        self.before_write = None
        self.updates = 0

    def update_one(self, query, update, upsert=False):
        self.updates += 1
        if query["_id"] not in self.documents and upsert:
            self.documents[query["_id"]] = {**query, **update["$setOnInsert"]}

    def find_one(self, query):
        return self.documents.get(query["_id"])

    def create_index(self, fields):
        self.indexes.append(fields)

    def bulk_write(self, operations, ordered):
        assert ordered is True
        self.batches.append(list(operations))
        if self.before_write is not None:
            callback, self.before_write = self.before_write, None
            callback()
        for operation in operations:
            key = operation.filter["_id"]
            if isinstance(operation, Delete):
                self.documents.pop(key, None)
            else:
                self.documents[key] = operation.replacement.copy()


@pytest.fixture
def mongo(monkeypatch, cfg):
    databases = {}
    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.admin = self
            self.closed = False
            self.options = kwargs
            clients.append(self)

        def command(self, name):
            assert name == "ping"

        def __getitem__(self, name):
            if name not in databases:
                databases[name] = SimpleNamespace(
                    meta=Collection(), articles=Collection(),
                    spark_runs=Collection(), daily_summaries=Collection(),
                )
            return databases[name]

        def close(self):
            self.closed = True

    monkeypatch.setattr(store, "_pymongo", lambda: SimpleNamespace(
        MongoClient=Client, ReplaceOne=Replace, DeleteOne=Delete
    ))
    live = Client()[cfg.settings.big_data.mongo_database]
    return SimpleNamespace(live=live, databases=databases, clients=clients)


def _insert(conn, text="Tamil Nadu farmers protest MSP delay", published=NOW):
    return db.insert_article(conn, prepare(RawHeadline(text, None, "News", published, "farmers")),
                             ingested_at=NOW)


def test_outbox_is_atomic_and_captures_all_data_changes(conn):
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            _insert(conn)
            raise RuntimeError("rollback")
    assert store.pending_articles(conn) == 0
    seed(conn, "Jobs and Unemployment", -0.4, NOW)
    first = conn.execute("SELECT revision FROM article_sync_outbox").fetchone()[0]
    assert first == 3  # article, classification, sentiment were committed together
    conn.execute("UPDATE classifications SET is_relevant = 0")
    conn.execute("UPDATE sentiment_scores SET compound = -0.6")
    conn.execute("DELETE FROM classifications")
    conn.execute("DELETE FROM sentiment_scores")
    assert conn.execute("SELECT revision FROM article_sync_outbox").fetchone()[0] == first + 4


def test_sync_is_idempotent_and_uses_utc_event_time(conn, cfg, mongo):
    _insert(conn, published=NOW + timedelta(days=1))
    assert store.sync_articles(conn, cfg, "live") == 1
    document = next(iter(mongo.live.articles.documents.values()))
    assert document["_id"] == document["headline_hash"]
    assert document["event_time"] == document["ingested_at"] == NOW
    assert document["published_at"] == NOW + timedelta(days=1)
    assert document["event_time"].utcoffset() == timedelta(0)
    assert document["is_relevant"] is False
    assert document["is_synthetic"] is False and document["dataset_kind"] == "live"
    assert store.pending_articles(conn) == 0
    assert store.sync_articles(conn, cfg, "live") == 0
    assert len(mongo.live.articles.documents) == 1


def test_reclassification_and_region_edits_replace_old_fields(conn, cfg, mongo):
    seed(conn, "Jobs and Unemployment", -0.4, NOW)
    store.sync_articles(conn, cfg, "live")
    conn.execute("UPDATE classifications SET is_relevant = 0, region_status = 'out_of_region'")
    store.sync_articles(conn, cfg, "live")
    document = next(iter(mongo.live.articles.documents.values()))
    assert document["is_relevant"] is False and document["region_status"] == "out_of_region"
    assert document["compound"] == -0.4  # retained audit score must not imply relevance
    conn.execute("DELETE FROM classifications")
    store.sync_articles(conn, cfg, "live")
    document = next(iter(mongo.live.articles.documents.values()))
    assert document["issue"] is None and document["method"] is None
    assert document["is_relevant"] is False


def test_failed_bulk_write_retains_queue_and_retries(conn, cfg, mongo):
    _insert(conn)

    def fail():
        raise RuntimeError("Mongo unavailable")

    mongo.live.articles.before_write = fail
    with pytest.raises(RuntimeError, match="Mongo unavailable"):
        store.sync_articles(conn, cfg, "live")
    assert store.pending_articles(conn) == 1
    assert store.sync_articles(conn, cfg, "live") == 1
    assert store.pending_articles(conn) == 0
    assert all(client.closed for client in mongo.clients[1:])


def test_acknowledgment_does_not_discard_a_newer_revision(conn, cfg, mongo):
    _insert(conn)
    mongo.live.articles.before_write = lambda: conn.execute("UPDATE articles SET source = 'Updated'")
    assert store.sync_articles(conn, cfg, "live") == 2
    assert next(iter(mongo.live.articles.documents.values()))["source"] == "Updated"
    assert store.pending_articles(conn) == 0


def test_tombstones_survive_article_id_reuse(conn, cfg, mongo):
    original_id = _insert(conn)
    store.sync_articles(conn, cfg, "live")
    original_hash = next(iter(mongo.live.articles.documents))
    conn.execute("DELETE FROM articles")
    assert _insert(conn, "Tamil Nadu electricity shortages worsen") == original_id
    store.sync_articles(conn, cfg, "live")
    assert original_hash not in mongo.live.articles.documents
    assert len(mongo.live.articles.documents) == 1
    assert store.pending_articles(conn) == 0


@pytest.mark.parametrize("batch_size", [1, 500])
def test_delete_and_recreate_same_hash_keeps_latest_document(conn, cfg, mongo, batch_size):
    cfg.settings.big_data.sync_batch_size = batch_size
    _insert(conn)
    store.sync_articles(conn, cfg, "live")
    conn.execute("DELETE FROM articles")
    _insert(conn)
    conn.execute("UPDATE articles SET source = 'Recreated'")
    store.sync_articles(conn, cfg, "live")
    assert next(iter(mongo.live.articles.documents.values()))["source"] == "Recreated"


def test_sync_batches_and_explicit_backfill_are_bounded(conn, cfg, mongo):
    cfg.settings.big_data.sync_batch_size = 2
    for n in range(5):
        _insert(conn, f"Tamil Nadu farmers protest MSP delay {n}")
    assert store.sync_articles(conn, cfg, "live") == 5
    assert [len(batch) for batch in mongo.live.articles.batches] == [2, 2, 1]
    mongo.live.articles.documents.clear()
    assert store.sync_articles(conn, cfg, "live", full=True) == 5
    assert len(mongo.live.articles.documents) == 5


def test_live_demo_separation_and_destination_stamp(conn, cfg, mongo, tmp_path):
    _insert(conn)
    with pytest.raises(db.DatasetKindError):
        store.sync_articles(conn, cfg, "demo")
    assert mongo.live.articles.documents == {}
    mongo.live.meta.documents["dataset_kind"] = {"_id": "dataset_kind", "value": "demo"}
    with pytest.raises(db.DatasetKindError):
        store.sync_articles(conn, cfg, "live")
    assert store.pending_articles(conn) == 1
    demo = db.init_db(tmp_path / "demo.db", "demo")
    try:
        article = prepare(RawHeadline("[DEMO] Tamil Nadu farm aid", None, "Synthetic", NOW, "farmers"))
        db.insert_article(demo, article, ingested_at=NOW, is_synthetic=True)
        assert store.sync_articles(demo, cfg, "demo") == 1
        destination = mongo.databases[cfg.settings.big_data.mongo_demo_database]
        assert next(iter(destination.articles.documents.values()))["is_synthetic"] is True
        assert mongo.live.articles.documents == {}
    finally:
        demo.close()


def test_recreated_sqlite_source_is_rejected(conn, cfg, mongo, tmp_path):
    _insert(conn)
    store.sync_articles(conn, cfg, "live")
    other = db.init_db(tmp_path / "other.db", "live")
    try:
        _insert(other, "Another journal")
        with pytest.raises(db.DatasetKindError, match="different SQLite journal"):
            store.sync_articles(other, cfg, "live")
        assert store.pending_articles(other) == 1
        assert len(mongo.live.articles.documents) == 1
    finally:
        other.close()


def test_dashboard_connection_performs_no_initialization(cfg, mongo):
    with pytest.raises(db.DatasetKindError):
        with store.open_mongo(cfg, "live"):
            pass
    assert mongo.live.meta.updates == 0 and mongo.live.articles.indexes == []
    assert mongo.live.spark_runs.indexes == [] and mongo.live.daily_summaries.indexes == []
    mongo.live.meta.documents["dataset_kind"] = {"_id": "dataset_kind", "value": "live"}
    with store.open_mongo(cfg, "live") as destination:
        assert destination is mongo.live
    assert mongo.live.meta.updates == 0 and mongo.live.articles.indexes == []
    assert mongo.live.spark_runs.indexes == [] and mongo.live.daily_summaries.indexes == []


def test_writer_initializes_indexes_for_latest_run_and_summary_queries(cfg, mongo):
    with store.open_mongo(cfg, "live", initialize=True):
        pass
    assert mongo.live.spark_runs.indexes == [[
        ("dataset_kind", 1), ("status", 1), ("completed_at", -1)
    ]]
    assert mongo.live.daily_summaries.indexes == [[
        ("run_id", 1), ("dataset_kind", 1), ("day", -1), ("issue", 1)
    ]]
    assert mongo.clients[-1].options["w"] == "majority"
    assert mongo.clients[-1].options["wTimeoutMS"] == cfg.settings.big_data.mongo_timeout_ms


def test_migration_backfills_existing_articles(tmp_path):
    conn = db.connect(tmp_path / "old.db")
    try:
        conn.executescript("BEGIN;" + db.MIGRATIONS[0] + db.MIGRATIONS[1] + "PRAGMA user_version=2;COMMIT;")
        conn.execute("INSERT INTO meta(key, value) VALUES ('dataset_kind', 'live')")
        _insert(conn)
        db.migrate(conn)
        assert store.pending_articles(conn) == 1
        assert conn.execute("SELECT value FROM meta WHERE key = 'article_sync_source'").fetchone()[0]
        db.migrate(conn)
        assert store.pending_articles(conn) == 1
    finally:
        conn.close()
