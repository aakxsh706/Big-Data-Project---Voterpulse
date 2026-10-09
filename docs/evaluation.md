# Evaluation plan

No evaluation results ship with this project. Every metric must be produced by
running the scripts below on headlines annotated by people. Results land in
`data/evaluation/results/<timestamp>_<kind>/`.

## 1. Build the 300-headline test set

1. Let the scheduler collect for at least a week (the sample is stratified across the
   six feeds and needs enough variety).
2. Export the sample:
   ```bash
   python scripts/evaluate.py sample --n 300
   ```
   This writes `data/evaluation/to_annotate.csv` with the columns of
   `data/evaluation/annotation_template.csv`. Headlines used as prompt examples are
   excluded automatically.
3. **Issue labels.** Two annotators independently fill `gold_issue` (exactly one of the 7
   labels in `config/issues.yaml`) and, optionally, `gold_secondary`. Disagreements are
   resolved by discussion; record the outcome in `issue_notes`. Annotate before
   looking at the model's predictions.
4. **Sentiment labels.** Three annotators independently fill
   `sentiment_annotator_1..3` with `Positive`, `Neutral` or `Negative`, judging the
   *wording of the headline*, not whether the news is good for them.
5. Save as `data/evaluation/annotated.csv`.

Annotation guidance: label the issue the headline is mainly about. Party infighting,
candidate lists and polling logistics are *Not an election issue*. When a headline
covers two issues, the primary is the one its main verb or subject concerns.

## 2. Classification evaluation

```bash
python scripts/evaluate.py classifier data/evaluation/annotated.csv --method both --repeat 3
```

Reports, for the LLM and the keyword baseline on the same rows:

* precision, recall and F1 for each of the 7 labels, with target flags (P ≥ 0.90, R ≥ 0.85);
* macro precision, recall and F1 over the six issues (undefined values are listed, not set to 0);
* confusion matrix (CSV);
* rejection of irrelevant headlines (recall and precision of *Not an election issue*);
* mean and p95 classification time;
* LLM–keyword agreement rate;
* **reproducibility:** with `--repeat 3`, the same headlines, model, prompt and settings are
  classified three times and every item whose label changed is listed.
  Temperature 0 is not assumed to be perfectly deterministic.

LLM failures (timeout, invalid output) are counted as errors, not silently replaced
by the fallback, so the LLM is measured on its own.

## 3. Sentiment evaluation

```bash
python scripts/evaluate.py sentiment data/evaluation/annotated.csv
```

* Gold label is the strict majority (at least 2 of 3). Items where all three annotators
  disagree are excluded and listed (`unresolved_item_ids`). Discuss them and either
  record a consensus label (and re-run) or report them as excluded.
* Agreement: Fleiss' kappa (all three annotators) and pairwise Cohen's kappa.
* VADER against the majority: accuracy, macro F1 and a confusion matrix.

## 4. Performance

```bash
python scripts/evaluate.py benchmark --n 50              # default placement
python scripts/evaluate.py benchmark --n 50 --cpu-only   # forces num_gpu=0
```

Records mean, median and p95 latency, total time, model load time, projected time for one
pipeline batch (`max_classifications_per_run`) and whether it fits in 15 minutes,
process and Ollama memory, system RAM, and CPU/GPU placement (from Ollama `/api/ps`).

**8 GB / no-GPU requirement:** run the `--cpu-only` benchmark on a machine with 8 GB of RAM
(or a VM limited to 8 GB) and no discrete GPU, while the dashboard is open. Record the
results table below. On Apple Silicon Macs the GPU is used by default, so always
include a `--cpu-only` run.

| Machine | RAM | CPU-only | Mean ms | p95 ms | Batch fits 15 min? | Ollama RSS MB | Date |
|---|---|---|---|---|---|---|---|
| *(fill in from benchmark.json)* | | | | | | | |

## 5. Reporting template

```
Test set: N=___ headlines annotated by ___ people between ___ and ___.
Issue annotation agreement (2 annotators, before discussion): ___ %.
LLM (model@digest ___, prompt ___): macro P ___, macro R ___, macro F1 ___.
  Issues meeting P≥0.90: ___/6. Issues meeting R≥0.85: ___/6.
Keyword baseline: macro P ___, macro R ___, macro F1 ___.
Irrelevant-headline rejection (LLM): recall ___, precision ___.
LLM–keyword agreement: ___ %.
Reproducibility: ___ % identical across ___ runs; ___ items changed.
Sentiment: Fleiss κ ___; VADER accuracy ___, macro F1 ___ (n=___, ___ excluded).
Performance: ___ ms/headline mean on ___ (CPU-only: yes/no); batch of ___ in ___ s.
Usability: see usability_study.md.
```
