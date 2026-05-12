# When LSTMs Beat a Small Transformer on Chinese Name Generation

*A fairness-first revisit of a small but revealing sequence-modeling problem.*

## TL;DR

I revisited an earlier Chinese name generation experiment under tighter control:

- fixed `train / val / test` split
- three random seeds
- parameter-matched small LSTM and small causal Transformer
- explicit Stage 1 and Stage 2 study design

The result stayed consistent:

> In this dataset and model-size regime, a parameter-matched small LSTM outperformed a parameter-matched small causal Transformer on Chinese micro-sequence generation.

At full data, the LSTM reached a mean test perplexity of `77.9209`, compared with `86.7124` for the Transformer. In a data-scale ablation, that direction stayed the same at `10%`, `30%`, and `100%` training data.

I do **not** take this as evidence that LSTMs are generally better than Transformers. I take it as evidence that **task geometry still matters**, and that short constrained generation can favor a different inductive bias than long-context language modeling.

## Why this small task is worth caring about

Chinese full names are short, but they are not structurally empty. A typical output is only two to four characters long, yet generation still depends on:

- surname format, including compound surnames
- local character compatibility
- stylistic regularities in historical data
- distributional bias in the source corpus

That makes the task a good example of a **micro-sequence** problem: the context window is tiny, but local structure still matters.

This is exactly the kind of setup where it is easy to over-generalize from modern NLP defaults. A lot of current sequence-model discussion is shaped by long-context tasks, large-scale pretraining, and attention-heavy architectures. But not every generation problem looks like document-scale language modeling. Some are short, local, and tightly structured.

I like this task because it is small enough to run carefully, but rich enough to expose real modeling tradeoffs.

## Why I redid the experiment

An earlier version of this project already pointed in an interesting direction: the LSTM seemed stronger than the small Transformer. But I was not comfortable publishing that conclusion in stronger language, because the setup still had some obvious weaknesses:

- the models were not tightly parameter-matched
- some conclusions leaned too heavily on limited runs
- the evaluation story was not yet as reproducible or structured as I wanted

So instead of trying to defend the old write-up harder, I narrowed the question and improved the protocol:

> Under stricter control, does an LSTM still outperform a small causal Transformer on this task?

That led to a fairness-first redesign with two completed stages:

- **Stage 1:** the full-data matched-pair comparison
- **Stage 2:** a data-scale ablation at `10%`, `30%`, and `100%`

I intentionally stopped there. I did **not** run the larger-capacity robustness pair from the longer protocol, because Stage 1 and Stage 2 already produced a sufficiently strong evidence chain for a careful public-facing claim.

## Dataset and preprocessing

The source data comes from the China Biographical Database (CBDB). In the current study pipeline:

- the cleaned dataset contains `584,442` full names
- the split is fixed at `80 / 10 / 10`
- the split seed is `20260419`
- the training seeds are `13`, `37`, and `73`

The project uses character-level tokenization with special symbols such as `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>`. In the full-data matched-pair runs, the shared training vocabulary is about `8.4k` tokens.

For the public repository, I do **not** treat the raw local CBDB database as a redistributable artifact. The public code documents the acquisition and preprocessing path, while excluding the raw database from the repo by default.

## Models

This project still includes a simple Markov baseline, but the main claim in this post comes from the **primary matched pair**:

| Model | Parameters |
| --- | ---: |
| LSTM-small-matched | `5,469,500` |
| Transformer-small | `5,355,708` |

That puts the two models within about `2.1%` of each other by parameter count, which makes the comparison much more defensible than the older version.

Conceptually:

- the **LSTM** carries a compact recurrent hidden state from step to step
- the **Transformer** is a small decoder-only causal model with learned positional embeddings and self-attention

Neither model is meant to simulate frontier-scale language modeling. The point here is not scale. The point is to compare architectural behavior on a short constrained generation problem.

## Training and evaluation protocol

The revised study uses a much cleaner experimental setup:

- fixed split reused across every run
- three random seeds per reported result
- best checkpoint selected by validation perplexity
- structured run artifacts and per-epoch history logs
- teacher-forced validation and test perplexity
- sample-based generation metrics on the best checkpoint

For the Stage 2 ablation, I retrained the same matched pair at:

- `10%` of the training split
- `30%` of the training split
- `100%` of the training split

The main Stage 2 comparison below uses a fixed sampling temperature of `0.8`.

I am not claiming that this reaches publication-grade rigor in every sense. I **am** claiming that it is dramatically better controlled than the original version and strong enough to support a portfolio-grade conclusion with appropriately careful wording.

## Stage 1: The main matched-pair result

The first question was simple: once the models are parameter-matched and evaluated across three seeds, does the full-data result still favor the LSTM?

Yes.

![Stage 1 test perplexity](results/fairness_v1/figures/stage1_primary_test_perplexity.png)

*Figure 1. Mean test perplexity across three seeds for the primary matched pair at full data. Lower is better.*

At full data and temperature `0.8`, the mean test perplexities were:

| Model | Mean test perplexity | Seed std. dev. |
| --- | ---: | ---: |
| LSTM (small matched) | `77.9209` | `0.0342` |
| Transformer (small matched) | `86.7124` | `0.4884` |

That is a gap of about `8.79` perplexity points in favor of the LSTM. More importantly, the result is not hanging on one lucky run: there is **no seed reversal**. The worst LSTM run still beats the best Transformer run.

That alone already makes the new study much stronger than the earlier draft.

## What the convergence curves add

The next obvious question is whether the Transformer simply had not trained long enough.

The answer appears to be no.

![Stage 1 convergence](results/fairness_v1/figures/stage1_primary_convergence.png)

*Figure 2. Mean train and validation perplexity across three seeds for the full-data matched pair.*

For the full-data runs:

- the LSTM reached its best validation perplexity around epoch `12`
- the Transformer needed roughly epoch `23-31` to reach its best validation perplexity
- even after longer training, the Transformer plateaued above the LSTM

This does not prove *why* the LSTM wins. But it does remove one easy explanation for the result. The Transformer is not losing merely because it was obviously undertrained.

## Stage 1: Distribution and constraint behavior

The story is not only about likelihood fit.

![Stage 1 distribution metrics](results/fairness_v1/figures/stage1_distribution_metrics.png)

*Figure 3. Full-data generation metrics for the primary matched pair at temperature `0.8`.*

At full data and temperature `0.8`:

- LSTM `Distinct-2`: `0.8851`
- Transformer `Distinct-2`: `0.8383`
- LSTM `unique_name_ratio`: `0.9805`
- Transformer `unique_name_ratio`: `0.9782`

So the LSTM is not just better on teacher-forced fit. It also keeps stronger character-bigram diversity in this setup.

But I do not want to flatten the result into "LSTM wins on everything," because that is not true. On some novelty-related generation behavior, the Transformer remains competitive:

- LSTM `novelty_ratio`: `0.4900`
- Transformer `novelty_ratio`: `0.5095`
- LSTM `train_overlap_ratio`: `0.5100`
- Transformer `train_overlap_ratio`: `0.4905`

That is a useful nuance. A better summary is:

> On the full-data matched pair, the LSTM achieves clearly better likelihood fit while also preserving stronger local diversity, whereas the Transformer is slightly more novelty-seeking at the same sampling temperature.

The constraint view stays clean:

![Stage 1 constraint metrics](results/fairness_v1/figures/stage1_constraint_metrics.png)

*Figure 4. Constraint-oriented metrics for the full-data matched pair.*

Duplicate-character rates and invalid-output rates remain very low for both models, and compound-surname compatibility is handled correctly in the current evaluation pipeline. So the main result is not being driven by obviously broken decoding behavior on one side.

## Stage 2: Does the result survive data scaling?

This was the most important follow-up.

If the LSTM only won at full data, the result would still be interesting, but easier to dismiss as a scale-specific artifact. So I reran the same matched pair at `10%`, `30%`, and `100%` of the training data.

The result stayed stable.

![Stage 2 data-scale perplexity](results/fairness_v1/figures/stage2_data_scale_perplexity.png)

*Figure 5. Test perplexity across data scales for the primary matched pair.*

| Training fraction | LSTM mean test perplexity | Transformer mean test perplexity | LSTM advantage |
| --- | ---: | ---: | ---: |
| `10%` | `93.4379` | `101.6190` | `8.18` |
| `30%` | `84.7673` | `96.9825` | `12.22` |
| `100%` | `77.9209` | `86.7124` | `8.79` |

Two facts matter here:

1. The direction of the result never flips.
2. The architecture gap does not disappear when the training set shrinks.

That makes the conclusion substantially stronger than the earlier single-stage version of the project.

## Stage 2: The more interesting part of the ablation

The data-scale ablation did more than confirm the headline result. It also changed how the two models behaved on generation metrics.

![Stage 2 distribution changes](results/fairness_v1/figures/stage2_data_scale_distribution.png)

*Figure 6. Distribution-level generation metrics across `10%`, `30%`, and `100%` training data.*

At `10%` and `30%` data, the LSTM is not only better in perplexity. It also shows:

- higher `Distinct-2`
- higher `unique_name_ratio`
- higher `novelty_ratio`
- lower `train_overlap_ratio`

At `100%` data, that novelty advantage narrows and slightly flips, even while the LSTM remains better on perplexity and `Distinct-2`.

This is the most interesting result in the whole study, in my view. It suggests that the architecture difference is not a single-axis story. As data increases, the tradeoff between **fit**, **memorization**, and **novelty-seeking** changes, but the LSTM advantage on the main fit metric persists.

The quality/validity picture remains stable too:

![Stage 2 quality and validity](results/fairness_v1/figures/stage2_data_scale_quality.png)

*Figure 7. Quality and validity metrics across data scales.*

Across all fractions, duplicate and invalid-output rates stay low for both models. So the Stage 2 result is not being carried by obviously pathological sampling behavior.

## What I think this means

I do **not** read this as a verdict on Transformers.

I read it as evidence that short, highly constrained, left-to-right generation can reward a compact recurrent state more than a small attention-only decoder at this scale.

Another way to say it is this:

> This result is about a short constrained sequence regime, not about NLP in general.

A few interpretations seem plausible:

- the task is short enough that a recurrent hidden state is an especially efficient inductive bias
- the small Transformer may need more capacity, different optimization choices, or a different scaling regime to close the gap
- architecture choice should follow task topology, not popularity

This is the kind of result I like in applied ML. Not a universal law. Not a hype-driven dunk. Just a clear reminder that architecture choice should be grounded in the structure of the problem.

## Why this matters for the engineering side of the project

One reason I kept pushing this project is that I wanted it to be more than "a few training scripts plus a blog post."

The current pipeline now has:

- a reusable fixed-split manifest
- structured run directories and versionable artifacts
- per-epoch histories that can regenerate plots directly
- separate training, evaluation, and summary surfaces
- artifact handling that can feed both research and product workflows

That matters for portfolio value. I do not only want to show that I understand tensor math. I want to show that I can package experimental work into a cleaner ML system that can eventually back an inference API and a deployable app.

## Product extension: fixed-token naming without overclaiming

The web demo now separates one common user request into two different modes.

In **Historical Pattern** mode, a user can provide a fixed character or two-character phrase, such as `飞`, `妃`, or `若虚`. The backend then uses constrained search over the deployed generators and reports how much support the cleaned historical corpus provides for that token and position. The important design choice is that the product does not say "ancient Chinese people would not use this character." It says something narrower: the current cleaned corpus has strong, weak, or no observed support for this request.

In **Creative Ancient-Style** mode, the goal is different. The user is no longer asking for corpus evidence; they are asking for controllable creative generation. The route is a template Fill-in-the-Middle decoder that conditions on the desired token, position, and optional seed, then completes the rest of the name. This mode is intentionally labeled as creative, not as historical reconstruction.

That split matters because it protects both user experience and research integrity. A low-support character should not make the app feel broken, but the app also should not pretend that a creative completion is evidence of historical naming practice.

## Limits of the current study

This write-up is much stronger than the original draft, but it still has clear limits:

1. The main claim comes from one matched size tier, not a full cross-capacity sweep.
2. Stage 3 was intentionally skipped, so there is no larger-pair robustness check yet.
3. Human evaluation is not included in the current version.
4. This is still a project-level study, not a formal publication pipeline.
5. The dataset is historical, so historical plausibility does not automatically imply modern naming quality.

Because of those limits, I am careful with the wording. I do **not** claim that this study proves LSTMs are better than Transformers in general. I claim that:

- under this controlled setup
- on this dataset
- at this model scale
- on this short constrained generation task

the LSTM performed better.

## Conclusion

Across a better-controlled comparison, the LSTM remained stronger than the small causal Transformer on this Chinese micro-sequence generation task.

That result held at:

- `10%` data
- `30%` data
- `100%` data

which makes the conclusion far more robust than the earlier version of the project.

If I continue this line of work, the most useful next steps are:

- a larger-capacity robustness pair
- lightweight blinded human evaluation
- deeper integration of the study artifacts with the deployed API and frontend demo

For now, though, I think the evidence is strong enough to support a careful public-facing claim:

> On this task, at this scale, the LSTM beat the small Transformer.
