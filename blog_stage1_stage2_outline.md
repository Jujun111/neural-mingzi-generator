# Stage 1 + Stage 2 Blog Outline

This outline is designed for the current evidence state of the project:

- Stage 1 completed
- Stage 2 completed
- Stage 3 intentionally skipped
- no human-eval results yet

It is meant to help you rewrite the public-facing blog without over-claiming.

## Working title options

1. When LSTMs Beat a Small Transformer on Chinese Name Generation
2. Fairness-First Revisit: LSTM vs. Small Transformer on a Chinese Micro-Sequence Task
3. A Better-Controlled Chinese Name Generation Study: LSTM vs. Small Transformer

## One-paragraph abstract draft

This project revisits a Chinese name generation comparison under tighter experimental control. Using a fixed train/validation/test split, parameter-matched small models, and three random seeds, I compared a character-level LSTM and a small causal Transformer on a short constrained generation task. Across the main matched pair, the LSTM consistently achieved lower validation and test perplexity than the Transformer at `10%`, `30%`, and `100%` training-data fractions. The result does not imply that LSTMs are universally better than Transformers. It suggests that for this dataset and this micro-sequence regime, recurrent inductive bias remains competitive and may be especially effective for short left-to-right structured generation.

## Recommended figure set

Use these figures from `results/fairness_v1/figures`:

- `stage1_primary_test_perplexity.png`
- `stage1_primary_convergence.png`
- `stage1_distribution_metrics.png`
- `stage2_data_scale_perplexity.png`
- `stage2_data_scale_distribution.png`
- `stage2_data_scale_quality.png`

## Blog structure

## 1. Why this task is interesting

Goal:
Explain why Chinese name generation is a useful micro-sequence problem rather than a toy.

Key points:

- Full names are only a few characters long, but they are still structured.
- The task rewards local compatibility, surname-conditioned generation, and short-range composition.
- This is a good place to test whether modern architecture defaults transfer cleanly to short constrained generation.

Suggested paragraph starter:

Chinese name generation is a useful example of a micro-sequence task: the output is short, but the structure is not trivial. A model still has to respect surname format, produce plausible character combinations, and avoid obviously broken local patterns.

## 2. Why I redid the experiment

Goal:
Show methodological maturity. Admit the old setup was interesting but not strong enough.

Key points:

- Earlier comparison was directionally interesting but not fairness-first.
- Old evidence mixed single-run results with non-matched models.
- The revised study fixes the split, matches parameter scale more closely, and uses three seeds.
- The new question is narrower and better posed.

Suggested sentence:

The earlier version of this project suggested that the LSTM was stronger than the small Transformer, but the evidence was not controlled tightly enough to support that claim confidently.

## 3. Experimental protocol

Goal:
Make the new setup sound clean, reproducible, and reviewer-friendly.

Include:

- fixed `80 / 10 / 10` split
- split seed `20260419`
- training seeds `13 / 37 / 73`
- primary matched pair:
  - `lstm_small_matched`
  - `transformer_small`
- parameter counts:
  - LSTM: `5,469,500`
  - Transformer: `5,355,708`
- selection rule:
  - best checkpoint chosen by validation perplexity
- Stage 2 design:
  - same pair retrained at `10%`, `30%`, `100%`
- evaluation temperature for Stage 2 comparison:
  - `0.8`

Good tone:

I am not claiming a perfect research-paper fairness standard. I am claiming that this setup is much more controlled than the original version and is strong enough for a careful portfolio-grade conclusion.

## 4. Stage 1: Main matched-pair result

Goal:
State the primary result clearly and modestly.

Lead figure:

- `stage1_primary_test_perplexity.png`

Supporting figure:

- `stage1_primary_convergence.png`

Numbers to report:

- full-data test perplexity at `T=0.8`
  - LSTM mean: `77.9209`
  - Transformer mean: `86.7124`
  - gap: about `8.79` perplexity points
- seed behavior:
  - no seed reversal
  - LSTM variance is very small

Suggested wording:

Under the primary matched pair, the LSTM consistently achieved lower validation and test perplexity than the small Transformer across all three seeds. The direction of the result was stable, not dependent on a single lucky run.

Interpretation note:

- do say:
  - in this dataset and model-size regime, the LSTM fit the task better
- do not say:
  - this validates LSTMs as a better sequence model in general

## 5. Stage 1: Distribution and constraint behavior

Goal:
Show that the result is not only about one fit metric.

Lead figures:

- `stage1_distribution_metrics.png`
- `stage1_constraint_metrics.png`

Numbers worth mentioning:

- full-data `Distinct-2`
  - LSTM mean: `0.8851`
  - Transformer mean: `0.8383`
- full-data `unique_name_ratio`
  - LSTM mean: `0.9805`
  - Transformer mean: `0.9782`
- full-data novelty / overlap:
  - LSTM novelty: `0.4900`
  - Transformer novelty: `0.5095`
  - LSTM overlap: `0.5100`
  - Transformer overlap: `0.4905`

Interpretation:

- The LSTM wins clearly on fit.
- It also keeps stronger character-bigram diversity.
- At full data, the Transformer is slightly more novelty-seeking and slightly less overlapping with the training set.
- That means the conclusion is not "LSTM is better on every axis."
- The defensible conclusion is that LSTM gives a better fit-quality tradeoff on this task, while the Transformer remains competitive on some novelty-oriented generation behavior.

## 6. Stage 2: Data-scale ablation

Goal:
Answer the most important follow-up question: does the result survive data scaling?

Lead figure:

- `stage2_data_scale_perplexity.png`

Supporting figures:

- `stage2_data_scale_distribution.png`
- `stage2_data_scale_quality.png`

Numbers to report:

- `10%` data
  - LSTM test perplexity: `93.4379`
  - Transformer test perplexity: `101.6190`
  - gap: about `8.18`
- `30%` data
  - LSTM test perplexity: `84.7673`
  - Transformer test perplexity: `96.9825`
  - gap: about `12.22`
- `100%` data
  - LSTM test perplexity: `77.9209`
  - Transformer test perplexity: `86.7124`
  - gap: about `8.79`

Main message:

The architecture gap does not disappear when the training set shrinks. The LSTM remains better at `10%`, `30%`, and `100%` data fractions, and the result stays directionally consistent across seeds.

Important nuance:

- At `10%` and `30%`, the LSTM is not only better in perplexity.
- It also shows:
  - higher `Distinct-2`
  - higher `unique_name_ratio`
  - higher `novelty_ratio`
  - lower `train_overlap_ratio`
- At `100%`, that novelty advantage narrows and slightly flips.

This is a strong paragraph to include:

The data-scale ablation makes the result more interesting, not less. The LSTM advantage is not a full-data-only effect, and the qualitative tradeoff between fit and novelty changes as the training set grows.

## 7. What I think the result means

Goal:
Offer a careful interpretation without sounding hand-wavy.

Suggested framing:

I do not read this as a verdict on Transformers. I read it as evidence that short, highly constrained, left-to-right generation can reward a compact recurrent state more than a small attention-only decoder at this scale.

Possible interpretation bullets:

- The task is short enough that recurrence may be an efficient inductive bias.
- The Transformer may need either more capacity, different optimization choices, or a different scaling regime to close the gap.
- Architecture choice should follow task geometry, not popularity.

## 8. Limits and non-claims

Goal:
Protect the blog from overstatement.

Keep these limits explicit:

- This is still a project-level study, not a formal publication.
- The main claim comes from one matched size tier.
- Stage 3 was not run, so there is no cross-capacity robustness check yet.
- Human evaluation is not included in the current write-up.
- The dataset is historical, so modern naming quality is still an open question.

Good sentence:

The current evidence is strong enough to support a careful claim about this experimental regime, but not strong enough to justify a universal conclusion about LSTMs and Transformers.

## 9. Portfolio / engineering note

Goal:
Connect the ML study to the software-engineering story.

Keep this short near the end:

- reproducible split and artifact layout
- structured training logs
- evaluation pipeline separated from demo flow
- FastAPI and frontend work exist as the productization path

Suggested sentence:

Part of the value of this project is that the research workflow and the deployment workflow now share a cleaner artifact structure, which makes the work easier to explain, reproduce, and extend.

## 10. Conclusion

Short version:

Across a better-controlled comparison, the LSTM remained stronger than the small causal Transformer on this Chinese micro-sequence generation task. That result held at `10%`, `30%`, and `100%` data scale, which makes the conclusion more robust than the original single-stage version of the project.

## Suggested caption drafts

`stage1_primary_test_perplexity.png`

Mean test perplexity across three seeds for the primary matched pair at full data. Lower is better.

`stage1_primary_convergence.png`

Mean train and validation perplexity across three seeds for the full-data matched pair. The LSTM reaches a lower validation plateau with fewer epochs.

`stage1_distribution_metrics.png`

Full-data generation metrics for the primary matched pair at temperature `0.8`. The LSTM keeps higher character-bigram diversity while the Transformer is slightly more novelty-seeking at full data.

`stage2_data_scale_perplexity.png`

Test perplexity across `10%`, `30%`, and `100%` training-data fractions. The LSTM remains better at every data scale tested.

`stage2_data_scale_distribution.png`

Distribution-level generation metrics across data scales. The architecture gap changes in magnitude, but not in direction, as more training data is added.

## Final writing checklist

- Keep "evidence in this setup" language.
- Use "matched pair" whenever you talk about the main claim.
- Avoid "validated", "proved", or "statistically significant".
- Mention Stage 3 was intentionally skipped rather than forgotten.
- Keep one sentence that explicitly says the result is about a short constrained sequence regime, not all NLP tasks.
