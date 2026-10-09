# Score-to-language thresholds

VoterPulse Lite's central job is turning numbers into short, cautious sentences. This
document explains every threshold in `config/sentiment_thresholds.yaml` and
`config/settings.yaml`, shows behaviour at the boundaries, and describes how to revise
the values with human evaluation. Every boundary listed here has a unit test in
`tests/test_translation.py` or `tests/test_sentiment.py`.

> The values below are **initial proposals**. They are reasoned, not empirically
> validated for Indian election headlines. Treat them as hypotheses to test with the
> procedure in the last section.

## 1. Headline label (one headline)

| Rule | Value | Why |
|---|---|---|
| Positive if compound ≥ | `0.05` | Cut-off recommended by VADER's authors (Hutto & Gilbert, 2014) and used in most VADER studies, so results are comparable with prior work. |
| Negative if compound ≤ | `-0.05` | Same source; symmetric. |
| Neutral otherwise | | Scores near zero mostly come from headlines with no lexicon words, which is common in factual reporting. |

Boundaries are inclusive: `0.05 → Positive`, `0.0499 → Neutral`, `-0.05 → Negative`.

Known weakness: one negative-lexicon word ("crime", "protest", "attack") can push a
factual headline below −0.05, so VADER tends to over-label political news as negative.
This is the main reason the aggregate rules below require more than a small mean.

## 2. Issue mood (many headlines)

Rules are applied in this order (`translator.classify_mood`):

1. **Not enough data** — fewer than `min_headlines = 25` headlines, or no mean.
2. **Mostly neutral** — at least `neutral_dominant_share = 60%` of headlines are neutral
   **and** the mean lies strictly between `negative_mean` and `positive_mean`.
3. **Mostly positive** — mean ≥ `positive_mean = 0.10` **and**
   positive share − negative share ≥ `share_margin = 10 points`.
4. **Mostly negative** — mean ≤ `negative_mean = -0.10` **and**
   negative share − positive share ≥ 10 points.
5. **Mixed** — everything else.

### Why these values

* **25 headlines minimum.** With n = 25 the standard error of a mean compound
  score (typical SD ≈ 0.4–0.5 for news headlines) is about 0.08–0.10. That is just
  small enough that a mean beyond ±0.10 is unlikely to be pure noise, while still
  being reachable for most issues within a week. Below 25, a few dramatic headlines
  can flip the label. The project specification also sets this value.
* **Mean ±0.10, twice the headline cut-off.** An average just past ±0.05 can come
  from one or two strong headlines among many neutral ones. Doubling the cut-off
  requires a consistent lean.
* **Share margin 10 points.** The mean alone can be dragged by a handful of extreme
  scores (for example "killed", "massacre"). Requiring that one side also *outnumbers*
  the other protects against "mostly negative" being driven by a few outliers. Both
  conditions must hold, so a conclusion needs agreement between the mean and the
  distribution.
* **Neutral 60%.** When most headlines carry no emotional wording, calling the
  coverage "mixed" would mislead, because nothing is really mixed. A strong mean
  (|mean| ≥ 0.10) overrides neutrality, because the non-neutral minority then leans
  clearly one way.

### Boundary examples (n = 30)

| Mean | Pos % | Neg % | Neu % | Result | Reason |
|---|---|---|---|---|---|
| 0.10 | 40 | 30 | 30 | Mostly positive | both conditions exactly met |
| 0.0999 | 40 | 30 | 30 | Mixed | mean just below |
| 0.20 | 40 | 30.01 | 29.99 | Mixed | margin just below 10 points |
| −0.10 | 30 | 40 | 30 | Mostly negative | both conditions met |
| −0.30 | 35 | 40 | 25 | Mixed | strongly negative mean but sides close |
| 0.05 | 30 | 10 | 60 | Mostly neutral | 60% neutral, weak mean |
| 0.05 | 30 | 10.01 | 59.99 | Mixed | neutral share just below 60% |
| any | any | any | any (n = 24) | Not enough data | sample size |

## 3. Weekly trend

* Compares the **last 7 complete IST days** with the **7 days before**, so windows are
  equal in length and today's partial day never distorts the comparison.
* |Δ mean| < `change_min = 0.05` → *broadly unchanged*. 0.05 is the smallest change
  that equals the width of VADER's own neutral band. Smaller shifts are within
  day-to-day noise for samples of 25–100 headlines.
* Each week needs `min_headlines_each_window = 25`, and the earlier week must lie
  entirely after collection started. Otherwise the trend is *not available*.

Boundaries: Δ = 0.05 → more positive, 0.0499 → unchanged, −0.05 → more negative.

## 4. Mood-shift alerts

| Setting | Value | Why |
|---|---|---|
| Current window | 24 h | "What changed today", as the specification requires. |
| Reference window | 7 days before the current window | A full weekly cycle, so weekday/weekend publishing patterns average out. No overlap with the current window. |
| Minimum headlines | 25 in each window | Same insufficient-data rule as the cards. |
| Trigger | |Δ| ≥ 0.15 | Three times the trend threshold. Alerts compare a short, noisy 24 h window, so they need a larger effect to avoid false alarms. |
| "Marked" severity | |Δ| ≥ 0.30 | Twice the trigger. |
| Reset | |Δ| < 0.075 | Half the trigger (hysteresis), so an alert doesn't flicker on and off around the threshold. |

## 5. Freshness

`stale_after_minutes = 60`. That is four missed 15-minute runs, long enough to ignore one
slow or failed run but short enough to warn users before they rely on old data.

## 6. Revising thresholds with human evaluation

1. **Collect judgements.** For about 60 issue-week summaries spread across all moods,
   show three non-technical annotators the actual headlines (not the numbers). Ask:
   "Overall, is this coverage mostly positive, mostly negative, mixed, or mostly
   neutral?"
2. **Build a reference.** Use the majority answer. Summaries with no majority are
   discussed and either resolved by consensus or excluded (record which).
3. **Compare.** Run the translator over the same summaries and compute agreement and
   a confusion matrix (reuse `voterpulse.evaluation.metrics`).
4. **Adjust one parameter at a time** (for example `share_margin` 0.10 → 0.15). Recompute
   on a *held-out* half of the summaries to avoid overfitting.
5. **Prefer caution.** When two settings perform similarly, choose the one that produces
   *Mixed* or *Not enough data* more often. A false "mostly negative" claim is worse
   than an uncertain one.
6. **Record every change** in the change log below, with date, old and new value,
   evidence, and the commit that updated the YAML and the tests.

### Change log

| Date | Parameter | Old | New | Evidence |
|---|---|---|---|---|
| — | (initial values) | — | as above | reasoning in this document; not yet validated |
