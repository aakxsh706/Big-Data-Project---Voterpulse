# Architecture

## Processes

Two independent processes share one SQLite file:

```
┌──────────────────────────── Process 1: collector ───────────────────────────┐
│ scripts/run_scheduler.py  (APScheduler, every 15 min, max_instances=1)      │
│ scripts/run_pipeline.py   (same pipeline, run once — manual/debug)          │
│                                                                             │
│  Google News RSS ──► rss_fetcher ──► normalizer ──► deduplicator            │
│        (6 feeds)      timeout+retry   clean+SHA-256   batch + UNIQUE hash   │
│                                                │                            │
│                                                ▼                            │
│                     classification/service.py: cache ─► Ollama LLM ─►       │
│                     validator (schema+evidence) ─► keyword cross-check      │
│                              │ fallback if down/invalid                     │
│                              ▼                                              │
│                 relevant? ──yes──► VADER ──► sentiment_scores               │
│                     │no                                                     │
│                     └──► stored for audit only                              │
│                                                                             │
│                     monitoring/alerts.py (24 h vs previous 7 days)          │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ writes (WAL, BEGIN IMMEDIATE, busy_timeout)
                                ▼
                     ┌─────────────────────┐
                     │ data/voterpulse.db  │   data/demo/voterpulse_demo.db
                     │ (SQLite, WAL)       │   (separate, stamped "demo")
                     └─────────┬───────────┘
                               │ reads only (plus human review decisions)
┌──────────────────────────────▼──── Process 2: dashboard ────────────────────┐
│ streamlit run app.py                                                        │
│  data_access ─► aggregator (windows, trend) ─► translator (plain language)  │
│  components / charts ─► Overview · Issue Explorer · Headlines ·             │
│                         Methodology · Developer Insights                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

The dashboard never starts a scheduler, so Streamlit reruns cannot create duplicate
jobs. Overlapping pipeline runs (scheduler plus a manual run) are prevented by an
atomic run lock in `pipeline_runs` (`database.start_run`). A run stuck for longer than
30 minutes is marked `abandoned`.

## Package layout

| Module | Responsibility |
|---|---|
| `config.py` | Loads and validates all YAML into Pydantic models; applies env overrides; defines IST. |
| `logging_utils.py` | `key=value` structured logs to console and a rotating file. |
| `models.py` | Shared dataclasses and enums (`ClassificationMethod`, `ValidationStatus`, …). |
| `database.py` | Connections (FK on, WAL, busy timeout), numbered migrations (`PRAGMA user_version`), parameterised queries, live/demo stamping. |
| `ingestion/` | RSS fetch and parse, headline cleaning and hashing, deduplication. |
| `classification/` | Prompt and schema generation, Ollama client, strict validator, keyword baseline, `service.py` orchestration (cache → LLM → validation → fallback). |
| `sentiment/` | VADER scoring, window aggregation, plain-language translator. |
| `monitoring/` | Pipeline orchestration, alerts, scheduler. |
| `dashboard/` | Read-side data access, Plotly charts, Streamlit components. |
| `evaluation/` | Metrics, annotation sampling, classifier and sentiment evaluation, benchmark. |

Additions to the suggested layout: `logging_utils.py` (shared logging),
`classification/service.py` (keeps fallback logic out of the reusable LLM client),
`evaluation/metrics.py` and `evaluation/sampling.py` (shared by several scripts), and
`pages/0_Overview.py` (with `app.py` as a small router, so the sidebar reads
"Overview").

## Database schema (migration 1)

```
articles 1──1 classifications           (UNIQUE article_id, CHECK secondary <> primary)
articles 1──1 sentiment_scores          (article_id is the PK → no duplicate scores)
articles 1──* review_queue              (UNIQUE (article_id, reason))
classification_cache                    (PK headline_hash, model_id, prompt_version)
pipeline_runs                           (one row per run, status + counters + errors)
alerts                                  (UNIQUE partial index: one active alert per issue)
meta                                    (dataset_kind = live | demo)
```

Foreign keys use `ON DELETE CASCADE`. Indexes cover publication and ingestion time,
issue/relevance, method, review status, run start time and sentiment label.

**Migrations:** append a new SQL string to `database.MIGRATIONS`. `migrate()` applies
the pending ones inside a transaction and bumps `user_version`. Never edit an applied
migration.

## Failure isolation

| Failure | Effect |
|---|---|
| One feed down or malformed | Logged and counted in `feeds_failed`; other feeds continue; run status `partial`. |
| All feeds down (offline) | Run status `failed`; the backlog is still classified; the dashboard shows old data with a freshness warning. |
| Ollama down | Keyword fallback, honestly labelled; circuit breaker avoids repeated timeouts. |
| Invalid LLM answer | Raw answer kept; keyword fallback; review queue entry. |
| Exception on one article | Rolled back, counted in `failed`, retried next run. |
| Crash mid-run | Run row becomes `abandoned` after 30 minutes; the next run proceeds. |
