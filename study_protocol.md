# Fairness-First Study Protocol

This document defines the deeper ablation workflow for the Chinese Name Generator project. It is designed to support a **paper-lite** level of evidence without pretending to be a full formal paper pipeline.

## Research question

Under stricter experimental control, does an LSTM still outperform a small causal Transformer on Chinese micro-sequence name generation?

## Experimental structure

### Main comparison

- `markov_baseline`
- `lstm_small_matched`
- `transformer_small`
- `lstm_large`
- `transformer_large_matched`

The main conclusion should be drawn from the **primary matched pair**:

- `lstm_small_matched`
- `transformer_small`

The larger pair is a robustness check:

- `lstm_large`
- `transformer_large_matched`

## Fixed defaults

- Split: `80 / 10 / 10`
- Split seed: `20260419`
- Fraction seed: `20260419`
- Training seeds: `13, 37, 73`
- Temperatures: `0.6, 0.8, 1.0`
- Prompt set: `李, 王, 张, 欧阳, 司马`
- Generation budget: `2000` unprompted samples per model/seed/temperature
- Prompted sample export: `250` samples total per evaluation run

## Budget and scheduling guidance

The following estimates are based on the current local environment:

- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- VRAM: `8 GB`
- Effective dataset size: about `584k` cleaned names
- Training split size: about `468k`
- Batch size: `128`
- Steps per full-data epoch: about `3653`

These are **planning estimates**, not guarantees. Early stopping, background load, and future code changes can move the final numbers.

### Approximate per-epoch training time

- `lstm_small_matched`: about `19s / epoch`
- `transformer_small`: about `27s / epoch`
- `lstm_large`: about `31s / epoch`
- `transformer_large_matched`: about `56s / epoch`

### Approximate full training time per run

If a run goes all the way to `40` epochs:

- `lstm_small_matched`: about `13 min`
- `transformer_small`: about `18 min`
- `lstm_large`: about `20 min`
- `transformer_large_matched`: about `38 min`

If early stopping triggers around `12-18` epochs, a more realistic range is:

- `lstm_small_matched`: about `4-6 min`
- `transformer_small`: about `6-8 min`
- `lstm_large`: about `6-9 min`
- `transformer_large_matched`: about `11-17 min`

### Approximate evaluation time per trained checkpoint

At the current default of `2000` unprompted samples plus `250` prompted samples:

- each `temperature` evaluation takes about `1.7-2.1 min`
- all three temperatures together take about `5-6.5 min` per trained checkpoint

This means the study is not training-only. The generation/evaluation sweep is a significant part of the wall-clock budget.

### Approximate total runtime for the default study

For `run-default-study`, a realistic planning range is:

- **fast case with earlier stopping**: about `4-4.5 hours`
- **slower case with many long runs**: about `6-7 hours`

That total includes:

- multi-seed training
- three-temperature generation/evaluation
- summary export

It does **not** include human scoring time.

### Recommended run order

Do not start with the full sweep unless you specifically want to leave it running unattended for hours.

Recommended order:

1. Run the **primary matched pair** at full data and all three seeds.
2. Inspect whether the main architecture gap is stable across seeds.
3. Run the `10%` and `30%` data-scale ablations for the primary pair.
4. Run the **large robustness pair** last.
5. Export human-eval sheets only after the automatic metrics are in place.

### Practical scheduling plan

If you want the fastest path to an evidence-backed blog update:

- **Stage 1: main claim only**
  - primary matched pair
  - full data
  - `3` seeds
  - expected runtime: about `1.2-2.2 hours`

- **Stage 2: data-scale ablation**
  - primary matched pair
  - `10%` and `30%`
  - `3` seeds
  - expected runtime: about `40-60 min`

- **Stage 3: robustness check**
  - large matched pair
  - full data
  - `3` seeds
  - expected runtime: about `1.3-2.1 hours`

This staged approach lets you stop early if the main matched-pair result is already unstable or unconvincing.

## Artifact layout

Each run is stored as:

`results/<run_id>/<model_spec>/seed_<seed>/split_<split_name>/fraction_<pct>/`

Training artifacts:

- `config.json`
- `history.csv`
- `best.ckpt`
- `final.ckpt`
- `metrics.json`

Evaluation artifacts:

- `eval/temperature_<T>/test_metrics.json`
- `eval/temperature_<T>/samples.jsonl`

Study-level artifacts:

- `summary/summary.csv`
- `figures/*.png`
- `human_eval/human_eval.csv`
- `human_eval/human_eval_key.csv`

## Recommended execution order

### 1. Create the fixed split

```bash
python study.py prepare-split --db-path latest.db --split-path data/cbdb_fixed_split_v1.json
```

### 2. Run the default fairness-first study

This runs:

- all five model variants at fraction `1.0`
- the primary matched pair again at fractions `0.1` and `0.3`
- all default temperatures
- all default seeds

```bash
python study.py run-default-study --split-path data/cbdb_fixed_split_v1.json --db-path latest.db
```

### 3. Export the summary table

```bash
python study.py summarize --split-path data/cbdb_fixed_split_v1.json
```

### 4. Plot the figures

```bash
python plot_study.py --run-id fairness_v1
```

### 5. Run the Transformer-tuned optimization check

Reviewer feedback identified the original flat-Adam Transformer recipe as a major confounder. Before making a strong architecture claim, run a tuned Transformer baseline with the same architecture as `transformer_small` but a Transformer-appropriate optimization recipe:

- model spec: `transformer_small_tuned`
- optimizer: `AdamW`
- peak learning rate: `5e-4` by default
- scheduler: linear warmup over the first `10%` of optimizer steps, then cosine decay
- dropout: `0.1`
- seeds: `13,37,73`
- data fraction: `100%`

```bash
python study.py run-transformer-tuned-baseline --run-id fairness_v1_transformer_tuned --split-path data/cbdb_fixed_split_v1.json --db-path latest.db
```

Optional higher-peak-LR check:

```bash
python study.py run-transformer-tuned-baseline --run-id fairness_v1_transformer_tuned_lr1e3 --split-path data/cbdb_fixed_split_v1.json --db-path latest.db --learning-rate 0.001
```

Interpretation rule: if the tuned Transformer closes or reverses the gap, the previous result should be reported as a training-recipe finding, not an architecture finding. If the LSTM still wins, the micro-sequence inductive-bias claim becomes much stronger.

### 6. Export human evaluation sheets

```bash
python study.py export-human-eval --split-path data/cbdb_fixed_split_v1.json --temperature 0.8
```

## Human evaluation rubric

Rate each candidate from `1` to `5` on:

- `naturalness`
- `realism`
- `phonetic_flow`
- `novelty_balance`

Scoring guidance:

- `1`: clearly weak or unnatural
- `3`: plausible but mixed
- `5`: strong and convincing

The CSV is blinded on purpose. Use `human_eval_key.csv` only after scoring is complete.

## Blog structure after experiments finish

The next version of the blog should follow this order:

1. Task definition and the micro-sequence hypothesis
2. Why the earlier evidence was not strong enough
3. How the new protocol improves experimental control
4. Main matched-pair results
5. Data-scale and temperature ablations
6. Human evaluation
7. Limits of the study and what still remains open

## Reporting rules

- Do not use interpolated curves as evidence.
- Do not present a single lucky seed as the headline result.
- Do not claim strong causality from a single metric.
- Use matched-pair results for the main architecture claim.
- Do not publish the LSTM-vs-Transformer claim without reporting the Transformer-tuned warmup/cosine check.
- Treat human evaluation as supporting evidence, not as a license to overclaim.
