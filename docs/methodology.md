# Methodology

> **VoterPulse Lite measures sentiment in news coverage, not public opinion or voting intention.**

## 1. Data collection

* **Source:** Google News RSS search, India edition in English (`hl=en-IN`, `gl=IN`,
  `ceid=IN:en`). There is one **Tamil Nadu** search per issue (`config/rss_feeds.yaml`),
  each limited to the last 7 days (`when:7d`). Queries are issue-focused (for example
  `("Tamil Nadu" OR Chennai) (unemployment OR jobs OR TNPSC …)`) rather than requiring the
  word "election". In testing, "election" queries returned mostly polling-procedure
  stories with no policy content.
* **Frequency:** every 15 minutes, from a separate scheduler process. Feeds are fetched
  one after another with a 2-second pause, a 15-second timeout and at most 3 retries
  (backoff of 2, 4 and 8 seconds). A failed feed is logged and skipped; the others continue.
* **Stored per headline:** the original headline, a normalised form, a SHA-256 hash,
  the publication name, link, publication time and ingestion time. All times are stored
  in UTC and shown in IST (UTC+05:30).
* **Cleaning:** HTML entities and tags are removed, whitespace is collapsed, and
  Google's trailing " - Publisher" suffix is removed.
* **Duplicates:** the hash is computed from a case-folded, punctuation-free form, so
  "Youth unemployment dominates rallies!" and "YOUTH Unemployment — dominates rallies"
  count as one headline. A UNIQUE constraint enforces this across all feeds and runs.
* **Search terms only retrieve candidates.** The classifier decides the issue.
* **Event time.** Aggregation uses the publication time. If the feed gives none, or gives
  a time later than ingestion (a clock error), the ingestion time is used instead.

Coverage depends entirely on Google News. It can change its results, rate-limit
requests or stop serving a feed at any time.

## 2. Issue classification

1. **Cache.** The cache is keyed by (headline hash, model + digest, prompt version).
   A cached, validated answer is reused and the LLM is not called again.
2. **Local LLM.** Llama 3.2 3B Instruct (Q4_K_M, Ollama tag `llama3.2:3b`) runs through
   Ollama's `/api/chat` with `temperature 0`, `seed 42` and a strict JSON schema in
   `format`. The system prompt is generated from `config/issues.yaml`: issue
   descriptions, rules, and examples of relevant and irrelevant headlines, including
   *"Power struggle in state cabinet"* labelled *Not an election issue*.
3. **Validation (Pydantic + rules).** Valid JSON only; exactly the keys
   `issue`, `evidence` and `secondary`; `issue` is one of the 7 labels; `secondary` is null or
   one of the 6 issues, differs from `issue`, and is null when `issue` is *Not an
   election issue*. Any violation means the answer is rejected (never silently fixed)
   and the raw text is kept for audit.
4. **Evidence check.** Each `evidence` span (spans are separated by "...") must
   appear in the headline, ignoring case, whitespace and quote style. If not, the
   label is kept, marked `evidence_unverified`, and queued for review.
5. **Keyword cross-check.** A deterministic keyword classifier
   (`keyword_classifier.py`) labels every headline too. If it disagrees with a valid
   LLM label, the **LLM label is kept** and the headline enters the review queue.
6. **Fallback.** If Ollama is down, times out after retries, or returns an invalid
   answer, the keyword label is used and stored with `method = keyword_fallback`.
   After 3 consecutive connection failures in a run, the LLM is skipped for the rest
   of that run (circuit breaker). `scripts/run_pipeline.py --reclassify-fallbacks`
   re-queues those headlines once Ollama is back.

7. **Region filter (Tamil Nadu).** The LLM prompt names the region, but tests showed the
   3B model often keeps headlines about other states. A deterministic rule therefore
   marks a headline `out_of_region` when it names a place from `region.other_places`
   (other states, union territories and big cities) and nothing from `region.local_terms`
   (Tamil Nadu districts, cities, state parties and agencies such as TNPSC and TNPDCL).
   Headlines naming no place stay in, because the searches already target Tamil Nadu.
   Out-of-region headlines keep their issue label for audit but are never scored or
   counted. `run_pipeline.py --recheck-region` re-applies edited lists.

Each classification stores the primary and secondary issue, evidence, method, keyword
label, model identifier with digest, prompt version, raw response, validation status,
latency and timestamp. **Only primary issues are aggregated**, so each headline counts
once.

## 3. Sentiment scoring

VADER (`vaderSentiment` 3.3.2) scores each **accepted** headline. Rejected headlines
are stored for audit but never scored or aggregated. Positive, negative, neutral and
compound scores are stored once per article (primary-key constraint). Label cut-offs
are ±0.05; see `thresholds.md`.

## 4. Aggregation

* Windows are half-open `[start, end)` in UTC: the last 24 hours, 7 days and 30 days
  (rolling, ending now), or custom IST calendar dates.
* A window is marked **partial** when collection began after the window started,
  or when a custom range extends into the future.
* Per issue the system computes the count, mean compound score, positive, neutral and
  negative counts and shares, sources, and latest headline time.
* **Weekly trend:** the last 7 complete IST days compared with the 7 days before.
* **Insufficient data:** fewer than 25 headlines gives a grey "Not enough data" label and no
  directional claim.

## 5. Plain-language translation

`sentiment/translator.py` turns the numbers into a mood label (with colour, icon and text),
a one-line statement ("Mostly negative, and getting worse."), explanation sentences, a
sample-size sentence ("Based on 86 news headlines collected over the last seven
days.") and a freshness sentence ("Updated 12 minutes ago."). Rules and thresholds:
`thresholds.md`.

Language policy: always "coverage" or "headlines", never "voters", "people" or "approval";
no exclamation marks or alarm words; sample size always shown; partial windows disclosed.

## 6. Mood-shift alerts

Each run compares the mean of the last 24 hours with the mean of the preceding 7 days,
per issue. An alert opens when |Δ| ≥ 0.15 and both windows have ≥ 25 headlines. A
unique partial index allows only one active alert per issue; while the shift
continues, that alert is updated rather than duplicated. It resolves when |Δ| < 0.075,
when the direction flips (a new alert opens if the flip exceeds the trigger), or when
the data becomes insufficient. Alerts are shown on the dashboard only; nothing is
sent externally.

## 7. Limitations

* News tone is not public opinion, and headline tone is not article tone.
* Google News favours large English-language outlets; regional and Indian-language
  media are under-represented, and one outlet can dominate an issue (the Issue Explorer
  shows the source concentration).
* VADER is a general-purpose English lexicon. It misreads sarcasm, negation patterns
  common in headlines ("crime falls"), Hindi or Hinglish terms, and policy vocabulary.
* A 3B-parameter model makes classification errors. The keyword cross-check and review
  queue catch some of them.
* Temperature 0 and a fixed seed reduce variation but do not guarantee identical output
  across Ollama versions, hardware or quantisations. Measure this with
  `evaluate.py classifier --repeat`.
* No data is collected while the computer is off or offline. The dashboard keeps showing
  older data and labels it as possibly out of date.
* No rankings, party comparisons or predictions are made.
