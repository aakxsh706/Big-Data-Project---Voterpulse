# VoterPulse Lite — Election Campaign Sentiment Tracking

A free, fully local dashboard that shows how six election policy issues in
**Tamil Nadu** are being **covered in the news**. It collects Tamil Nadu headlines from
Google News every 15 minutes, sorts them into issues with a local open-weight LLM, scores their wording with
VADER, and turns the numbers into short, cautious statements:

> **Jobs and Unemployment**
> Mood: Mostly negative, and getting worse.
> Based on 86 news headlines collected over the last seven days. Updated 12 minutes ago.

> **VoterPulse Lite measures sentiment in news coverage, not public opinion or voting intention.**

There are no paid APIs, no cloud LLMs and no API keys. Internet access is needed only to
fetch RSS feeds and to install packages and the model.

---

## Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Installation (Windows / macOS / Linux)](#installation)
4. [Running](#running)
5. [Demo mode](#demo-mode)
6. [Tests](#tests)
7. [Evaluation](#evaluation)
8. [Configuration](#configuration)
9. [Project structure](#project-structure)
10. [Verification status](#verification-status)
11. [Limitations](#limitations)
12. [MongoDB + Apache Spark](#mongodb--apache-spark)

## How it works

```
Google News RSS (Tamil Nadu searches) → clean & hash → de-duplicate → local LLM (Ollama, Llama 3.2 3B)
  → keyword cross-check (fallback if the LLM is unavailable) → region filter → VADER sentiment
  → SQLite → aggregation → plain-language translation → Streamlit dashboard + alerts

Big-data profile: SQLite change queue → MongoDB articles → Apache Spark (parallel VADER + daily aggregation)
  → MongoDB daily summaries → Developer Insights / Big Data
```

* Six issues plus *Not an election issue* are defined in `config/issues.yaml`. The LLM prompt,
  JSON schema and keyword baseline are all generated from that file.
* Every LLM answer is validated against a strict schema, and its quoted evidence words
  must appear in the headline. Invalid answers are never accepted silently.
* When the LLM is unavailable, the keyword classifier is used and labelled
  `keyword_fallback`. Disagreements between the LLM and the keywords go to a review
  queue, and the LLM label is kept.
* Rejected headlines are stored for audit but never counted.
* **Region scope (Tamil Nadu).** The searches target Tamil Nadu, and the LLM is told the region.
  A deterministic filter also drops headlines that name another state or city (for example
  "Bihar" or "Bengaluru") and nothing in Tamil Nadu; the matched place is stored. The lists
  are `region` in `config/settings.yaml`. After editing them, run
  `python scripts/run_pipeline.py --recheck-region`. Set `region.name: null` and use national
  queries in `config/rss_feeds.yaml` for all-India coverage.
* Fewer than 25 headlines gives **"Not enough data"** in grey.
* Collection runs in its **own process**. The dashboard only reads, so it can never
  start duplicate schedulers.

Details: [docs/architecture.md](docs/architecture.md) ·
[docs/methodology.md](docs/methodology.md) · [docs/thresholds.md](docs/thresholds.md)

## Requirements

* Python **3.11 or newer**
* [Ollama](https://ollama.com) (free) and about 2.5 GB of free disk space for the model
* 8 GB RAM recommended; no GPU required. Measure your own machine with the benchmark
  (see [Evaluation](#evaluation)); this has not yet been verified on an 8 GB machine.
* For the big-data profile: MongoDB 8.0 (Docker Compose included), a **Java 17 JDK**, and
  the additional dependencies in `requirements-big-data.txt`.

## Installation

All commands are run from the project root folder (`voterpulse-lite/`).

### 1. Python virtual environment and dependencies

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

### 2. Ollama and the model

| OS | Install Ollama |
|---|---|
| macOS | Download from https://ollama.com/download, or `brew install ollama`, or unpack the standalone build `ollama-darwin.tgz` from the Ollama GitHub releases and run `./ollama serve` |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Windows | Download the installer from https://ollama.com/download |

Then (any OS):
```bash
ollama pull llama3.2:3b
ollama list            # note the ID (digest) shown for llama3.2:3b
```
Make sure the Ollama app or service is running (`ollama serve` if it isn't). The configured
digest is in `config/model.yaml` (`expected_digest`). If `ollama list` shows a
different ID, a different build was downloaded. The app logs a warning and records the
actual `model@digest` with every classification. To pin your build, update
`expected_digest`.

### 3. Initialise the database

```bash
python scripts/init_db.py
```

The app also creates the database automatically on first start.

## Running

Use **two terminals**, with the virtual environment activated in both.

**Terminal 1 — collector (every 15 minutes, runs once immediately):**
```bash
python scripts/run_scheduler.py
```

**Terminal 2 — dashboard:**
```bash
streamlit run app.py
```
Then open http://localhost:8501.

**Run the pipeline once (manual / debugging):**
```bash
python scripts/run_pipeline.py                         # RSS + LLM (keyword fallback if Ollama is down)
python scripts/run_pipeline.py --no-llm                # keyword classifier only
python scripts/run_pipeline.py --reclassify-fallbacks  # redo fallback / invalid rows with the LLM
python scripts/run_pipeline.py --recheck-region        # re-apply region lists to stored headlines
```

Each run classifies at most `max_classifications_per_run` (default 120) headlines, oldest
first, so a CPU-only batch fits inside the 15-minute interval. The rest waits for the
next run, and the pending count is shown on the Developer page.

**Offline:** the dashboard keeps showing stored data and labels it
"Data may be out of date". Runs while offline are recorded as `failed`.

## Demo mode

Demo mode populates a **separate** database with clearly synthetic headlines (each one
prefixed `[DEMO]` and attributed to fictional "(synthetic)" outlets). It needs neither the
internet nor Ollama. The dashboard shows a prominent "DEMO DATA" banner. The demo
database is stamped `demo` and the code refuses to mix it with live data.

```bash
python scripts/generate_demo_data.py
```
macOS / Linux:
```bash
VOTERPULSE_DEMO_MODE=1 streamlit run app.py
```
Windows PowerShell:
```powershell
$env:VOTERPULSE_DEMO_MODE="1"; streamlit run app.py
```
Demo mode is off by default. Unset the variable (or set it to `0`) to return to live data.
Demo headlines are classified by the keyword rules (recorded as `keyword_fallback`; no LLM
is involved) and scored by VADER. **Demo data must never be used as evaluation evidence.**

## Tests

```bash
pytest
```
The base tests are deterministic: no external network access and no real Ollama (HTTP is faked).
Spark transformation tests run on local workers when PySpark and Java are installed;
otherwise they are skipped. Mongo synchronization tests use a fake driver.
They cover ingestion, dedup, retries, LLM validation and fallback, caching, the keyword
baseline, VADER boundaries, aggregation windows, every translation threshold boundary,
alerts (trigger, duplicates, hysteresis, direction flip), the end-to-end pipeline, the
evaluation metrics, and headless rendering of every dashboard page with an empty
database and with demo data.

## Evaluation

All metrics come from your own human annotations; none are bundled. See
[docs/evaluation.md](docs/evaluation.md) and
[docs/usability_study.md](docs/usability_study.md).

```bash
python scripts/evaluate.py sample --n 300                                     # export headlines to annotate
python scripts/evaluate.py classifier data/evaluation/annotated.csv --repeat 3  # LLM vs keyword + reproducibility
python scripts/evaluate.py sentiment  data/evaluation/annotated.csv           # VADER vs 3 annotators
python scripts/evaluate.py benchmark --n 50 --cpu-only                        # latency, memory, CPU-only
```

## Configuration

| File | Contents |
|---|---|
| `config/issues.yaml` | Issue names, descriptions, keywords, exclusions, prompt examples |
| `config/rss_feeds.yaml` | Google News search queries, locale, timeouts, retries |
| `config/model.yaml` | Ollama host, model tag, expected digest, timeout, retries, options (temperature 0) |
| `config/sentiment_thresholds.yaml` | Headline, mood and trend thresholds (see docs/thresholds.md) |
| `config/settings.yaml` | DB paths, demo flag, logging, pipeline interval and batch size, aggregation and alert settings |

Environment overrides (or a `.env` file — see `.env.example`):
`VOTERPULSE_DB_PATH`, `VOTERPULSE_DEMO_MODE`, `VOTERPULSE_OLLAMA_HOST`,
`VOTERPULSE_MODEL`, `VOTERPULSE_LOG_LEVEL`, `VOTERPULSE_BIG_DATA_ENABLED`,
`VOTERPULSE_MONGO_URI`, `VOTERPULSE_MONGO_DATABASE`, `VOTERPULSE_MONGO_DEMO_DATABASE`,
`VOTERPULSE_MONGO_TIMEOUT_MS`, `VOTERPULSE_SYNC_BATCH_SIZE`,
`VOTERPULSE_SPARK_MASTER`, `VOTERPULSE_SPARK_PARTITIONS`.

Logs: console and `data/logs/voterpulse.log` (rotating), as `key=value` lines.

## Project structure

```
app.py                     Streamlit entry point (navigation)
pages/                     Overview · Issue Explorer · Headlines · Methodology · Developer Insights
config/                    YAML configuration
src/voterpulse/
  config.py database.py models.py logging_utils.py
  ingestion/               rss_fetcher, normalizer, deduplicator
  classification/          prompts, llm_classifier (Ollama), validator, keyword_classifier, service
  sentiment/               vader_analyzer, aggregator, translator
  monitoring/              pipeline, alerts, scheduler
  dashboard/               data_access, components, charts
  evaluation/              metrics, sampling, evaluate_classifier, evaluate_sentiment, benchmark
  big_data/                MongoDB article synchronization + Apache Spark batch analytics
scripts/                   init_db, run_pipeline, run_scheduler, generate_demo_data, evaluate, run_big_data
tests/                     pytest suite + fixtures
docs/                      architecture, methodology, thresholds, evaluation, usability_study
data/                      SQLite databases, logs, evaluation files (git-ignored except templates)
```

## Verification status

Verified during development on macOS (Apple Silicon), Python 3.13:

* The full pytest suite passes.
* Live Google News RSS retrieval: 6 of 6 feeds fetched, headlines stored, cross-feed
  duplicates rejected.
* Ollama unavailable: the circuit breaker opens and the keyword fallback is labelled honestly.
* All dashboard pages render with an empty database and with demo data, including
  in a real browser at desktop and phone widths.

* Live classification by `llama3.2:3b` through Ollama 0.40.2 (macOS, Apple Silicon): digest
  `a80c4f17acd5` confirmed with `ollama list`, structured JSON output accepted, about 0.7–1 s
  per headline on the default (GPU) placement. 320 of 321 real headlines produced valid
  answers; the one invalid answer fell back to keywords and is labelled accordingly.

**Not verified yet:**

* Classification accuracy against human labels, CPU-only speed on an 8 GB laptop, memory
  use and repeat-run consistency. Run `scripts/evaluate.py` to measure them.
* Windows and Linux. The code uses only portable paths and libraries but has not been
  run on those systems.
* Long-running scheduler behaviour over days.

## Limitations

* News tone ≠ public opinion; headline tone ≠ article content.
* Google News coverage is incomplete and favours large English-language outlets.
  **Tamil-language media are not collected** (VADER scores English only), so much of
  Tamil Nadu's local press is missing; a few sources can dominate.
* The region filter only catches headlines that *name* another place; an out-of-state
  story that names no place can still be counted.
* VADER is a general English lexicon: sarcasm, negation ("crime falls"), Hinglish
  and policy terms are misread.
* A 3B model makes mistakes; the review queue catches some, not all.
* Temperature 0 does not guarantee identical outputs across versions or hardware.
* Thresholds are reasoned initial values, not yet validated with users
  (see docs/thresholds.md).
* No data is collected while the machine is off; the dashboard says when data is stale.
* No party rankings, approval ratings or predictions, by design.

## MongoDB + Apache Spark

The big-data profile stores denormalized article documents in **MongoDB** and reads them
directly with the MongoDB Spark Connector. **Apache Spark** distributes VADER sentiment
scoring and daily issue aggregation across partitions. Results and run metrics are stored
in MongoDB and shown under **Developer Insights → Big Data**.

SQLite remains the operational journal for RSS collection, LLM classifications, reviews,
alerts, and the existing Overview/Issue Explorer pages. This is an additional batch
analytics path; those existing pages still use their local aggregation logic.

### Start the services

Install a Java **17 JDK**, set `JAVA_HOME` to its installation directory, and verify
`java -version`. Then, in the activated virtual environment:

```bash
pip install -r requirements-big-data.txt
docker compose up -d mongodb
```

Docker runs MongoDB only. Spark runs on the host using PySpark and Java, using multiple
local CPU cores by default (`local[*]`). The first Spark run needs internet access to
download the pinned connector JARs. MongoDB data persists in the Compose named volume.

Copy `.env.example` to `.env` if needed, retaining any existing settings, and set:

```dotenv
VOTERPULSE_BIG_DATA_ENABLED=1
VOTERPULSE_MONGO_URI=mongodb://127.0.0.1:27017
VOTERPULSE_SPARK_MASTER=local[*]
VOTERPULSE_SPARK_PARTITIONS=4
```

Restart the collector/dashboard after changing environment settings. With this flag on,
each collector run sends pending article changes to MongoDB. Spark batches run separately:

```bash
python scripts/run_pipeline.py --no-llm  # collect/classify + synchronize MongoDB
python scripts/run_big_data.py          # synchronize existing/pending articles, then run Spark
streamlit run app.py
```

### Demonstrate it with synthetic data

```bash
python scripts/generate_demo_data.py --days 90
python scripts/run_big_data.py --demo --explain
```

Set `VOTERPULSE_DEMO_MODE=1` for the dashboard to view the demo results. The job's explicit
`--demo` flag selects `voterpulse_demo`; without it the job selects the live `voterpulse`
database. Synthetic data stays separate and is never evaluation evidence.

The Big Data tab shows the successful batch's input count, accepted count, partition count,
Spark version, duration, and daily sentiment results. `--explain` prints Spark's execution
plan; the Spark UI at `http://localhost:4040` is available while the job runs. Local mode
demonstrates parallel execution on one machine. It does not demonstrate a multi-machine
cluster or establish production-scale throughput.

### Maintenance commands

```bash
python scripts/run_big_data.py --sync-only    # update MongoDB without starting Java/Spark
python scripts/run_big_data.py --skip-sync    # process the corpus already stored in MongoDB
python scripts/run_big_data.py --full-sync    # resend all articles to an empty/new MongoDB destination
docker compose stop mongodb                 # retain stored data
```

The SQLite outbox persists failed writes for retry, including reclassifications, region
changes, and deletions. Existing articles are queued during migration. A MongoDB outage
marks the collector run `partial` and leaves collected data available locally. Failed
Spark jobs never replace the last successful dashboard result. Details, cluster setup,
data contracts, and limitations: [docs/big_data.md](docs/big_data.md).
