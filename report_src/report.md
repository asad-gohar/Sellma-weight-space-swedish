---
title: "Is Swedish degeneration in a small open model a removable obstacle?"
subtitle: "Testing the weight-space-programming claim for SE.LLMA, candidate assignment report"
author: "Asad Gohar"
date: "November 2026"
---

## Summary

I tested the mechanism behind the article's headline LLM result in a Swedish setting. That mechanism is a sparse, targeted weight edit that removes a failure mode hiding an ability the model already has. The model was Qwen2.5-0.5B-Instruct, evaluated on held-out Swedish prompts.

1. **Language choice is fine; the visible failure is looping.** The model answers Swedish prompts in Swedish 93% of the time, but 78% of its Swedish answers loop at 96 tokens, against 23% of English ones.
2. **The article-style edit's effect is not established against random edits of the same size.**
   - The edit removes a learned loop direction from 64 `down_proj` columns (0.012% of weights).
   - At the dev-selected, overshooting strength it cuts loops by 14 points [3, 25] against baseline.
   - But against 15 random edits with the same update norm, it ranks second (empirical p = 0.125, i.e. (1+1)/(15+1)). Random edits on their own cut loops by about 6 points on average, so the direction-specific part (about 8 points) is suggestive, not established.
   - The literal removal (s = 1) has no significant effect.
3. **Decoding fixes remove most loops.** Sampling cuts them by 64–70 points over three seeds; the one graded seed shows no significant fluency cost. A repetition penalty cuts them by 75 points but lowers fluency.
4. **Removing loops doesn't reveal usable Swedish.**
   - In every arm, blind-judged adequacy is 0.07–0.15 out of 2, and none of 40 prompts gets a usable answer.
   - Loop-free answers score a little higher than looping ones (0.15 against 0.08), but none of 155 is fully adequate.
   - The same model answers the same 50 questions in English far better: 1.24 against 0.10 out of 2, with sampling that equalises looping across languages (1.12 against 0.06 with greedy decoding).
   - The deficit is Swedish competence, not a hidden ability.
5. **Mixed-language data can create a fake fix.** A loop direction built from mixed-language data "fixed" loops by switching the model out of Swedish.

**Recommendation:** for models that lack Swedish competence, don't pursue weight editing as a route to better Swedish. Keep the obstacle-versus-competence triage. The open question is whether edits help *competent* Swedish models with specific obstacles, and that is the next experiment.


## 1. What the article does and doesn't show

*Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab; MIT CSAIL, 2 Nov 2026) reports that GPT-6 Astra can read, repair and write network weights. The experiments come in two regimes that the conclusions should keep apart.

**A. Reasoning about weights with no forward passes.**

- Naming the game behind a 17k-weight policy (game name withheld), and the shape behind small diffusion networks.
- Identifying the language and the copying (induction) heads of 7M-weight character transformers.
- The one-shot repair of the Kinetix car policy.
- The one-shot MinAtar policies.
- The Go policy at zero forward passes (a 336-line Python program writing the weights), which won more than half its games against two small PPO networks.

**B. Edits found with repeated model feedback.**

- Go with up to 50 passes.
- The Qwen3.5-0.8B math edit, with 10, 25 or 50 passes. Each pass is a batch averaging 64 sequences (max 512).

### Strongest evidence (regime A)

- **Experiment 3** is the cleanest. Astra could inspect the weights in Python but could not run the network. The ground truth came from running the network on random repeated strings, which Astra never saw. The copying heads it named match that ground truth (copy scores 0.22–0.55 against ≤0.02 for every other head), with no false positives.
- **The MinAtar networks,** written with no data, rollouts or gradients, beat PPO at 100M steps in all three games. The method reads as compiling a hand-designed algorithm into weights: in Breakout the first layer computes where the ball will land, and the second moves the paddle there. That is a different skill from reading weights that were learned.

### Most important limitations

1. **The headline LLM result is in regime B.**
   - Gains appear at every budget, but the large jump needs 50 passes. MATH-500 goes 7.9 → 12.3 / 14.1 / 48.1 at 10 / 25 / 50 passes; GSM8K and IFEval already roughly double at 10.
   - The evaluation benchmarks were held out from Astra. But each budget is a single Astra run with no variance reported, so the 25 → 50 jump can't be told apart from run-to-run luck in a feedback-driven search.
2. **Reading learned behaviour from a real LM's weights failed in the one test of it.** In Experiment 7 (which of two Gemma 3 270M fine-tunes is sycophantic), weights alone failed in all four attempts. Astra succeeded once it could see attention patterns on texts it chose, which is activation-based interpretability rather than reading weights.
3. **The Qwen gain removes a failure rather than adding a skill.**
   - 93% of base answers hit the 32k-token limit, against 33% after the edit.
   - On Gemma 3 270M, which rarely loops (8.5%), no budget helped (MATH-500 6.8 → 7.0–7.7).
   - The authors draw the same conclusion: an edit can remove an obstacle but cannot add a missing ability.
4. **Controls missing for the headline claim.**
   - No random edit of the same size (131,072 weights in MLP layers 15 and 19).
   - No fine-tuning or steering baseline.
   - Single runs, no confidence intervals.
   - Decoding is partly covered. Base and edited models used the same anti-repetition sampling (presence penalty 1.5), and the base model still hit the limit 93% of the time, so decoding alone did not fix the looping in that setting. The authors also argue, fairly, that turning thinking off avoids the loops without fixing the weights. Whether an edit is worth having therefore depends on whether cheap decoding fixes already work in the setting at hand, and that has to be checked case by case.
5. **Narrow scope.** One frontier model, small networks the authors built themselves, and no comparison with human interpretability work.

The authors' "Response to Comments" (12 Nov) clarifies that "natively" means the LLM reasons about weights itself, using Python as a tool rather than an optimiser, and that weight-space programming "is not a method". The article's logs also credit a Hugging Face engineer with rebuilding the edited checkpoint from the logs. That supports the edit's reproducibility, but doesn't address the missing controls.

### What's justified

- Frontier LLMs can pull structure out of the weights of small networks and can hand-author small networks.
- On a real LLM, a sparse edit found *with model feedback* can remove a dominant failure mode and unlock an existing ability.

### What isn't justified (yet)

- That LLMs understand the weights of *LLMs* by reading them.
- That the Qwen edit came from reading weights rather than from feedback-guided search.
- That edits beat fine-tuning.
- That the effect generalises across models.

### What this means for SE.LLMA
The useful question is not "can an LLM read a Swedish model's weights?". It is "when a model is weak in Swedish, is the weakness a removable obstacle, which an edit might fix, or missing competence, which only data and training can fix?". That decides where SE.LLMA should spend its effort.


## 2. Hypothesis, and how it changed

**Why it matters for SE.LLMA.** SE.LLMA adapts open models to Swedish. Before spending compute, it helps to know which kind of problem a weak model has:

- an *obstacle*: a specific failure (looping, wrong language, refusing) that hides competence the model already has, which could be fixed cheaply and transparently; or
- *missing competence*, which only data and training will fix.

The article's Qwen result (loops removed, math unlocked) and its Gemma non-result (nothing to remove, nothing gained) are the two sides of this question.

**Original plan.** Fix *wrong-language* answers with a weight-only edit.

**Pivot.** Made before any test-set *run*, and confirmed on dev data: 97% of dev answers were in Swedish, while 73% of Swedish dev answers looped, against 0% of English ones at 96 tokens. A 10-prompt sanity check run earlier (6 of its 9 Swedish prompts and its 1 English prompt fall in the test split) had already shown the same pattern and led me to raise generation length from 64 to 96 tokens. Its outputs are saved in `results/extra/sanity_check_64tok.jsonl`.

**H.** Swedish degeneration in Qwen2.5-0.5B-Instruct is largely a removable obstacle.

**Supported if:**

- a targeted sparse edit reduces looping more than size-matched random edits, without regressions; and
- removing loops raises judged Swedish quality toward the model's English quality. English quality is measured on the 50 paired questions: 37 are test prompts and 13 dev prompts, and 14 are among the 40 graded test prompts. Only the baseline model answers them.

**Challenged if:**

- the edit doesn't beat size-matched random edits or decoding fixes; or
- quality stays flat once loops are gone.

## 3. Methods

**Model and decoding.**

- Model: Qwen2.5-0.5B-Instruct, the fp16 GGUF from Ollama (`qwen2.5:0.5b-instruct-fp16`, sha256 `6f96e01a…`), converted to fp32. The fp16 → fp32 step is lossless; Ollama's bf16 → fp16 export may round values. Hugging Face was not reachable from the compute environment.
- 24 layers, d = 896, 14 query heads, 2 KV heads, tied embeddings, 494M parameters.
- Hardware: 2 CPU cores.
- Decoding: greedy, 96 new tokens, the default Qwen system prompt.

**Data.**

- 110 Swedish prompts in five categories and 50 English prompts. The English prompts are translations of 50 of the Swedish ones; a few add context, such as "(allemansrätten)".
- All written for this study and split with seed 1234 into dev (30 sv / 15 en) and test (80 sv / 35 en) before any model run.
- Europarl sample from NLTK: files 0–5 are used to build directions and token sets; files 7–9 are held out for NLL.

**Metrics.**

- *Answer language:* lingua over sv/en/da/nb/nn/de/nl/zh.
- *Loop:* some word 5-gram occurs twice in the answer. This is crude and surface-level.
- *Regressions:*
  - English prompts still answered in English;
  - mean token NLL on 150 held-out Europarl sentences each in Swedish, English and Danish;
  - Danish/Norwegian leakage.
- *Quality:* blind 0–2 grades for fluency and adequacy. Each sheet went to a fresh LLM judge that saw shuffled items with no arm names. The verbatim prompts are in `results/judging/judge_prompts.md`.
  - Sheet A2: one judge, every arm, 40 test prompts.
  - Sheet B2: the 50 Swedish/English pairs, baseline model, 256 new tokens, greedy.
  - Sheet B3: the same pairs with sampling, which equalises loop rates across languages.
  - Earlier grading rounds (A, B, C) are kept for transparency and to measure agreement.

**Edits.** All edits are applied in place and reverted exactly afterwards.

- *Loop edit (article-style, uses feedback).*
  1. In the baseline answers to the 30 **Swedish** dev prompts, label each response token as looping if the token 8-gram ending there has already occurred: 571 looping and 2,091 non-looping tokens.
  2. Take d = mean MLP output on looping tokens minus mean on non-looping tokens, at layer L.
  3. Pick the 64 columns j of W_down with the largest |dᵀ W[:, j]| and set W[:, j] ← W[:, j] − s · d dᵀ W[:, j]. s = 1 removes the d-component; s = 3 flips it to −2× (an overshoot).
  4. Search L ∈ {6, 12, 15, 18, 21, 15+19} × s ∈ {1, 3} on dev. Select the lowest Swedish loop rate subject to at least 90% Swedish answers and the English-answer rate within 0.10 of baseline.
  5. Selected: L = 15, s = 3 (57,344 weights, update Frobenius norm 3.27). Its dev margin over L = 15, s = 1 was one prompt out of 30.
- *Random edits.* The same procedure with a random unit direction, at the same layer and strength. The generator seed is 1000·k + 15.
  - First version (k = 0–2): the same number of weights touched, but update norms of only 1.28–1.36.
  - Final version: **rescaled to the loop edit's update norm (3.27)**. k = 0–2 ran on the full test set; k = 3–14 on the 80 Swedish test prompts only, giving a null distribution of 15.
- *Activation steering (conventional).* Subtract β·Δh_L from the residual stream after block L, where Δh_L is the raw mean residual difference between looping and non-looping Swedish tokens. Six dev configurations; selected L = 18, β = 1.
- *Logit-lens heuristic edit (no feedback on behaviour).*
  1. Direction: g ⊙ (mean W_U[sv] − mean W_U[en]) over the top 300 Swedish and English Europarl tokens by frequency log-ratio.
  2. Written into layer 16's `v_proj` bias by least squares through W_o. A bias on V adds a constant vector to the residual stream, because attention weights sum to 1; that's 128 values.
  3. Only part of the requested vector is reachable: the achieved vector has cosine 0.40 with it.
  4. Scale: 0.5 × the norm of the *mean* residual vector at layer 16 (66.9, from 50 unlabeled sentences).
- *Decoding and prompting.*
  - Repetition penalty 1.1, 1.2 or 1.3, tuned on dev (selected 1.2).
  - Sampling: T = 0.7, top-p 0.8, repetition penalty 1.05 and top-k off, over three seeds. These are Qwen's recommended settings except that its top-k of 20 was *not* applied. Tables label this arm `qwen_recommended_sampling`.
  - A Swedish system prompt ("Svara alltid på svenska").

**Feedback budget.**

- Loop edit: 12 dev evaluations of 45 prompts each, plus 12 in the first, confounded search.
- Steering: 6 evaluations, plus 2 in the first search.
- Repetition penalty: 3 + 3.

## 4. Results

Configurations selected on dev: loop edit `loopedit_L15_s3.0` (57,344 weights), repetition penalty 1.2, steering `steer_L18_b1.0`; logit-lens edit at layer 16 (128 bias values, reach cosine 0.40). In the tables, `weight_only_sv_direction_W0` is the logit-lens heuristic edit, `random_edit_seed*` are the first random controls (matched in weights touched only) and `random_edit_normmatched_seed*` the final ones (also matched in update norm).


Loop edit selected on dev: `loopedit_L15_s3.0`, 57,344 weights; update Frobenius norm 3.27. Unmatched random updates had norm 1.28, 1.36, 1.31.

### 4.1 Automatic metrics (80 Swedish, 35 English test prompts; greedy unless stated)

| arm | sv: in Swedish | sv: Danish/Norwegian | sv: loop rate | Δ loop vs baseline [95% CI] | identical to baseline | en: in English | en: loop | ΔNLL sv / en / da |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.93 | 0.06 | 0.78 | – | 115/115 | 1.00 | 0.23 | 0 (4.240 / 3.751 / 4.476) |
| sv_system_prompt | 0.96 | 0.03 | 0.78 | +0.00 [-0.12, +0.12] | 2/115 | 0.69 | 0.37 | – |
| qwen_recommended_sampling | 0.89 | 0.07 | 0.07 | -0.70 [-0.81, -0.59] | 1/115 | 0.97 | 0.09 | – |
| repetition_penalty_1.2 | 0.89 | 0.10 | 0.03 | -0.75 [-0.84, -0.66] | 0/115 | 1.00 | 0.00 | – |
| loop_edit[loopedit_L15_s3.0] | 0.93 | 0.06 | 0.64 | -0.14 [-0.25, -0.03] | 15/115 | 1.00 | 0.23 | +0.008/+0.006/+0.006 |
| random_edit_seed0 | 0.93 | 0.05 | 0.76 | -0.01 [-0.07, +0.04] | 86/115 | 1.00 | 0.26 | -0.001/+0.000/-0.000 |
| random_edit_seed1 | 0.91 | 0.07 | 0.72 | -0.05 [-0.12, +0.01] | 63/115 | 1.00 | 0.26 | +0.000/-0.000/-0.000 |
| random_edit_seed2 | 0.93 | 0.06 | 0.72 | -0.05 [-0.12, +0.03] | 77/115 | 1.00 | 0.26 | -0.000/-0.000/+0.000 |
| weight_only_sv_direction_W0 | 0.84 | 0.11 | 0.84 | +0.06 [-0.04, +0.16] | 0/115 | 0.97 | 0.23 | +0.566/+0.408/+0.636 |
| steering[steer_L18_b1.0] | 0.86 | 0.10 | 0.62 | -0.15 [-0.29, -0.01] | 3/115 | 1.00 | 0.23 | +0.054/+0.061/+0.075 |
| qwen_recommended_sampling_seed1 | 0.91 | 0.07 | 0.07 | -0.70 [-0.79, -0.60] | 1/115 | 1.00 | 0.09 | – |
| qwen_recommended_sampling_seed2 | 0.90 | 0.05 | 0.14 | -0.64 [-0.75, -0.51] | 1/115 | 1.00 | 0.09 | – |
| random_edit_normmatched_seed0 | 0.90 | 0.07 | 0.78 | +0.00 [-0.09, +0.09] | 54/115 | 1.00 | 0.20 | -0.001/+0.001/+0.000 |
| random_edit_normmatched_seed1 | 0.90 | 0.09 | 0.75 | -0.03 [-0.11, +0.06] | 40/115 | 1.00 | 0.23 | +0.000/-0.000/+0.000 |
| random_edit_normmatched_seed2 | 0.91 | 0.07 | 0.62 | -0.15 [-0.26, -0.04] | 45/115 | 1.00 | 0.26 | -0.000/-0.001/+0.001 |
| loop_edit_s1[loopedit_L15_s1.0] | 0.91 | 0.07 | 0.71 | -0.06 [-0.16, +0.04] | 44/115 | 1.00 | 0.26 | +0.001/+0.001/+0.001 |

### 4.2 Targeted edit vs random edits (paired, 80 Swedish prompts; Δ loop rate)

- loop_edit[loopedit_L15_s3.0] − random_edit_seed0: -0.12 [-0.24, -0.01]
- loop_edit[loopedit_L15_s3.0] − random_edit_seed1: -0.09 [-0.20, +0.04]
- loop_edit[loopedit_L15_s3.0] − random_edit_seed2: -0.09 [-0.20, +0.03]
- loop_edit[loopedit_L15_s3.0] − random_edit_normmatched_seed0: -0.14 [-0.25, -0.01]
- loop_edit[loopedit_L15_s3.0] − random_edit_normmatched_seed1: -0.11 [-0.23, +0.00]
- loop_edit[loopedit_L15_s3.0] − random_edit_normmatched_seed2: +0.01 [-0.11, +0.14]
- loop_edit[loopedit_L15_s3.0] − mean of norm-matched random edits: -0.08 [-0.17, +0.02]
- mean of norm-matched random edits − baseline: -0.06 [-0.13, +0.01]
- loop_edit_s1[loopedit_L15_s1.0] (removal, s=1) − baseline: -0.06 [-0.16, +0.04]

Null distribution: 15 random directions, each rescaled to the loop edit's update norm. Swedish loop rates: min 0.625, median 0.725, max 0.775 (baseline 0.775). Loop edit 0.637: 1 of 15 random edits did as well or better; empirical one-sided p = 0.125 = (k+1)/(N+1).

Sampling arm over 3 seeds: Swedish loop rate 0.075, 0.075, 0.138 (mean 0.096); in-Swedish rate 0.89, 0.91, 0.90.

### 4.3 Blind quality grading, one judge for all arms (Sheet A2: 40 Swedish test prompts per arm)

Scores 0–2. Fluency: grammatical, natural Swedish. Adequacy: correct, relevant, helpful. One fresh LLM judge (judge 4) graded every item, shuffled, with no arm names.

| arm | fluency [95% CI] | adequacy [95% CI] | Δ fluency vs baseline (paired) | Δ adequacy vs baseline (paired) | usable answers (adequacy 2) |
|---|---|---|---|---|---|
| baseline | 0.90 [0.80, 0.97] | 0.12 [0.03, 0.23] | – | – | 0/40 |
| random_edit_normmatched | 0.88 [0.75, 0.97] | 0.15 [0.05, 0.28] | -0.03 [-0.07, +0.00] | +0.03 [+0.00, +0.07] | 0/40 |
| loop_edit | 0.90 [0.80, 0.97] | 0.15 [0.05, 0.28] | +0.00 [-0.07, +0.07] | +0.03 [-0.07, +0.12] | 0/40 |
| loop_edit_s1 | 0.93 [0.85, 1.00] | 0.15 [0.05, 0.28] | +0.03 [+0.00, +0.07] | +0.03 [+0.00, +0.07] | 0/40 |
| qwen_sampling | 0.82 [0.70, 0.93] | 0.10 [0.03, 0.20] | -0.07 [-0.23, +0.07] | -0.03 [-0.12, +0.05] | 0/40 |
| reppen | 0.72 [0.57, 0.85] | 0.07 [0.00, 0.17] | -0.17 [-0.30, -0.05] | -0.05 [-0.15, +0.05] | 0/40 |
| steering | 0.78 [0.62, 0.93] | 0.10 [0.03, 0.20] | -0.12 [-0.25, +0.00] | -0.03 [-0.07, +0.00] | 0/40 |
| random_edit_normmatched_s2 | 0.90 [0.80, 0.97] | 0.07 [0.00, 0.17] | +0.00 [+0.00, +0.00] | -0.05 [-0.12, +0.00] | 0/40 |
| sv_system_prompt | 1.05 [0.95, 1.15] | 0.10 [0.03, 0.20] | +0.15 [+0.03, +0.28] | -0.03 [-0.10, +0.05] | 0/40 |

### 4.3b First grading round (Sheet A: 6 arms, judge 1; Sheet C: sampling arm, judge 3)

Kept for transparency. Different judges graded different arms here, which Sheet A2 fixes.

| arm | fluency [95% CI] | adequacy [95% CI] | Δ fluency vs baseline (paired) | Δ adequacy vs baseline (paired) | usable answers (adequacy 2) |
|---|---|---|---|---|---|
| baseline | 0.93 [0.82, 1.02] | 0.10 [0.03, 0.20] | – | – | 0/40 |
| loop_edit | 0.95 [0.85, 1.05] | 0.07 [0.00, 0.17] | +0.03 [+0.00, +0.07] | -0.03 [-0.12, +0.07] | 0/40 |
| random_edit | 0.90 [0.78, 1.02] | 0.07 [0.00, 0.17] | -0.03 [-0.07, +0.00] | -0.03 [-0.07, +0.00] | 0/40 |
| sv_system_prompt | 0.97 [0.90, 1.05] | 0.10 [0.03, 0.20] | +0.05 [-0.07, +0.20] | +0.00 [-0.07, +0.07] | 0/40 |
| reppen | 0.75 [0.60, 0.88] | 0.05 [0.00, 0.12] | -0.17 [-0.33, -0.05] | -0.05 [-0.15, +0.05] | 0/40 |
| steering | 0.78 [0.65, 0.90] | 0.05 [0.00, 0.12] | -0.15 [-0.28, -0.05] | -0.05 [-0.12, +0.00] | 0/40 |
| qwen_sampling | 0.88 [0.78, 0.97] | 0.07 [0.00, 0.15] | -0.05 [-0.17, +0.07] | -0.03 [-0.07, +0.00] | 0/40 |

Agreement between the A2 judge and the first-round judges on 240 identical items: fluency 0.94, adequacy 0.95 (exact match).
Agreement of judge 3 re-grading 40 Sheet-A items: fluency 1.00, adequacy 0.97. Exact agreement is inflated by the floor effect: most items score fluency 1 and adequacy 0.

Baseline answers that loop: mean adequacy 0.13 (n=30); answers that don't loop: 0.10 (n=10).
Pooled over all A2 arms: answers that loop 0.08 (n=205), answers that don't 0.15 (n=155), 0 of the 155 loop-free answers fully adequate.

### 4.4 Cross-lingual gap: the same 50 questions in English and Swedish, baseline model, 256-token budget (Sheet B2, judge 5)

Answers at 256 new tokens: mean 119 words in Swedish and 164 in English. Loop rate: Swedish 0.86, English 0.72.

- Mean adequacy: English 1.12, Swedish 0.06; difference +1.06 [+0.88, +1.24] (n=50 pairs)
- Mean fluency: English 2.00, Swedish 0.72
- 1 of the 14 questions answered fully adequately in English were also adequate in Swedish

### 4.4a Cross-lingual gap with sampling, loop rates equal across languages (Sheet B3, judge 6)

Sampling (T=0.7, top-p 0.8, repetition penalty 1.05, top-k off), 256 tokens. Loop rate: Swedish 0.46, English 0.44; mean words 109 / 176.

- Mean adequacy: English 1.24, Swedish 0.10; difference +1.14 [+0.92, +1.36] (n=50 pairs)
- Mean fluency: English 2.00, Swedish 0.32
- 2 of the 21 questions answered fully adequately in English were also adequate in Swedish

### 4.4b First version (Sheet B, judge 2, 96-token budget; Swedish answers about a third shorter)


- Mean adequacy: English 1.32, Swedish 0.08; difference +1.24 [+1.04, +1.44] (n=50 pairs)
- Mean fluency: English 2.00, Swedish 0.80
- 1 of the 23 questions answered fully adequately in English were also adequate in Swedish


![Left: share of the 80 Swedish test answers that loop, per arm. Right: blind-judged adequacy (0–2) on 40 Swedish test prompts, one judge for all arms. The dashed line is the baseline model on the 50 paired questions asked in English (256-token answers), so it covers a different, overlapping prompt set.](../results/fig_main.png){width=100%}

### 4.5 Examples

Held-out test; the first ~200 characters of each answer, unedited.

*Prompt:* „Vad är skillnaden mellan väder och klimat?“

- **baseline** (loop=True): Skillnaden mellan väder och klimat är en av de mest viktigaste och viktiga problemer som man kan ha med resursen. Detta är en del av en viktig och viktigaste problemet som man kan ha med resursen, vil…
- **qwen_recommended_sampling** (loop=False): Skilnaden mellan väder och klimat är ett stor problem, men det finns en viktig fråga som du förstår inte:  "Vad är skillnaden mellan väder och klimat?"  Detta är en viktigt fråga om att berätta på en …
- **repetition_penalty_1.2** (loop=False): Skillnade mellan väders och klima var en av de mest viktigaste problemer i moderna planeten. Detta skulle vara ett allmänligt problem om det inte skulle kunna beklagar sig.  1. Klimaatverkstid: Värdet…
- **loop_edit** (loop=False): Skillnaden mellan väder och klimat är en av de mest viktigaste och viktiga problemer som man kan ha med resursen. Detta är en del av en viktig och viktiga räkningsproblemm, som kan ha en effektiv och …
- **random_edit_normmatched_seed2** (loop=False): Skillnaden mellan väder och klimat är en av de mest viktigaste och viktiga problemer som man kan ha med resursen. Detta är en del av en viktig och viktigaste världskrav, som är en viktig del av den gl…
- **steering** (loop=False): Skiljningarna mellan väder och klimat är en av de mest viktiga och viktiga värdena i dagens kulturen och kulturelltjänst. Detta kan vara en viktiga värde för att förstå och förbättra kulturelltjänsten…
- **weight_only_sv_direction_W0** (loop=True): Vad är skillnaden mellan väder och klimat är en fenomen som är en del av ett kritisk tidsproces. Detta är en kritisk tidsproces som kan ha ett störst eller en störst kritisk tidsproces på ett kritisk …

*Prompt:* „Vad är inflation?“

- **baseline** (loop=False): Inflation är en ekonomisk problem som kan vara viktigt för olika sätt. Detta är en av de mest viktigaste ekonomiska problemer i moderna tid. Inflation kan vara viktiga för olika sätt:  1. Ekonomisk: I…
- **qwen_recommended_sampling** (loop=False): Inflation är en faktor som kan vara viktiga för många personer i dag. Inflation är ett av de mest kända faktorer i ekonomi och det är en allmän problem som kan ändra lämplig många aspekter av ekonomin…
- **repetition_penalty_1.2** (loop=False): Inflation är en ekonomisk problem som kan vara viktigt för olika sätt. Detta har ett stor antal konsekvenser på många sätt.  1. Influenster: Inflation kan inbilda att försöka uppmärksamhet och tillgån…
- **loop_edit** (loop=True): Inflation är en ekonomisk delvis deltagande som kan vara viktiga för olika sätt. Det är en av de mest viktigaste delvis deltagande i ekonomi och det är en del av de mest viktigaste delvis deltagande i…
- **random_edit_normmatched_seed2** (loop=True): Inflation är en ekonomisk konstellation som kan vara viktiga för att förbättra och uppdatera demokrati och demokratiske system. Inflation är en förändring i kostnaden för den sammanlagande kostnaden a…
- **steering** (loop=True): Inflation är en av de mest viktiga och viktiga problemer som man kan ha med innehåll i dag. Det är en av de mest viktiga problemer som man kan ha med innehåll i dag, och det är en av de mest viktiga p…
- **weight_only_sv_direction_W0** (loop=True): Inflation är ett kvalitetsspektrum där det är ett kritiskt förväntad att få ett nytt kvalitetsspektrum. Det är en kritisk förväntad att få ett nytt kvalitetsspektrum är att få ett nytt kvalitetsspektr…

*Prompt:* „Hur fungerar det svenska pensionssystemet i stora drag?“

- **baseline** (loop=True): Det finns en stor varmhet och stor konfliktering mellan det svenska pensionssystemet och andra system som har en stora drag. Det finns en stor del av det svenska pensionssystemet som är en stora drag …
- **qwen_recommended_sampling** (loop=False): Det finns ett alltid svårare och förvångsfullt system än en enkelt, lösbar system som en svenska pensionssystemet. Det är det stora drag att förbättra och utveckla detta systemet så att det kan hålla …
- **repetition_penalty_1.2** (loop=False): Det finns en stor varmhet och problem som har kvar på svenska pensionssystemet:  1. Försäljning: Pensioner kan föra till 70-85 år med ett förslag av 24% av de övriga betalningar.  2. Kostnaden: Den öv…
- **loop_edit** (loop=True): Det finns en stor varmhet och stor konfliktering mellan svenska pensionssystemet och andra system som har en stora drag. Det finns en stor del av svenska systemet som är en större och större del av sy…
- **random_edit_normmatched_seed2** (loop=True): Det finns en stor varmhet och stor konfliktering mellan det svenska pensionssystemet och andra system som har en stora drag. Det finns en stor del av det svenska pensionssystemet som är en stora drag …
- **steering** (loop=False): Det finns ett stort råd som har tillgång till en önskad förhållande för pensionerade personer i svenska länder. I det svenska systemet, som är en av de mest populära och önskadte intryckna för pension…
- **weight_only_sv_direction_W0** (loop=True): Det är en kritisk och konseptuell problem för det svenska pensionssystemet. Detta är en system som är utkastat på ett större stället än på ett stället. Detta är en system som är utkastat på ett större…


## 5. Interpretation

**Does the targeted edit beat random edits? Not established.**

- At the selected strength (s = 3) the loop edit cut loops by 14 points [3, 25] against baseline, with negligible collateral damage (ΔNLL ≤ 0.01) and English unchanged.
- Against the first three random edits rescaled to the same update norm, it won twice and lost narrowly once (seed 2: 0.625 against the edit's 0.637). The pooled difference was −8 points [−17, +2], an interval that ignores variation between seeds.
- Against all 15 size-matched random edits (loop rates 0.625–0.775, median 0.725), the edit (0.637) ranks second: 14 looped more, and one looped slightly less. The empirical one-sided p = 0.125, i.e. (1+1)/(15+1). Random edits on their own cut loops by about 6 points on average. So roughly 40% of the edit's 14-point effect is generic perturbation, and the direction-specific part, about 8 points, is suggestive but not established.
- The literal reading of the article's "subtracted that direction" (s = 1) gave −6 points [−16, +4]. That is no significant effect, and about the size of an average random edit.
- **Caveat on the control.** Even at equal update norm, random edits change fewer outputs than the loop edit (44–56 of the 80 Swedish answers, against 70). Equal norm isn't equal functional size, so a stricter control would match the share of outputs changed or the KL divergence.

My first random control matched only the number of weights touched. Its updates were about 2.5× smaller, and most of its outputs were identical to baseline. A review of the first draft caught this. The comparisons against those controls were similar (−12, −9 and −9 points), but they weren't a fair null.

**Is the failure a removable obstacle? No.**

- Every arm that reduced loops left judged Swedish adequacy at 0.07–0.15 out of 2. No arm differs from baseline by more than 0.05, and no arm produced a usable answer.
- Pooled over all graded arms, answers that loop score lower (0.08, n = 205) than answers that don't (0.15, n = 155). Looping does cost something, though partly by construction, since the rubric tells judges to penalise loops. But none of the 155 loop-free answers is fully adequate. (In the baseline alone the difference goes the other way, 0.13 against 0.10, but with only 10 loop-free answers.)
- On the same 50 questions with 256-token answers, the model reaches:
  - with greedy decoding: 1.12 adequacy in English, 0.06 in Swedish;
  - with sampling, which equalises looping across languages (44% against 46%): 1.24 against 0.10, difference +1.14 [+0.92, +1.36].

  Swedish answers are still shorter (109 against 176 words), but that can't plausibly explain a gap of this size.
- Answers freed of loops stay confused. With sampling, photosynthesis is still "a technique … in flora and workplaces". With the repetition penalty, machine learning becomes the invented "Maskinlärgren (masking)".

The knowledge is often there in English, but the model can't express it in Swedish. This is the article's Gemma case, not its Qwen case.

**Cheap alternatives.**

- Sampling removed most loops (7–14% remaining over three seeds). The graded seed showed no significant fluency cost.
- The repetition penalty removed nearly all of them, but measurably lowered fluency (−0.17 [−0.30, −0.05]).
- In the article, both arms already used a strong presence penalty and 93% of the base model's answers still hit the 32k-token limit. Decoding alone wasn't enough there, and here it is. So I don't claim that decoding generally beats edits, only that it has to be checked first, in each setting.
- At 256 tokens English loops too (72%, against 86% for Swedish). Looping is a general failure of greedy decoding in this model; it just appears sooner in Swedish.

**Logit-lens heuristic edit.** A direction taken from the unembedding geometry and written into the network with no behavioural feedback made every metric worse:

- Swedish answers 93% → 84%;
- loops 78% → 84%;
- Swedish NLL +0.57 nats.

This is a hand-written heuristic in a single, a-priori configuration, not an LLM reading weights, so it doesn't test the article's capability. The scale may be to blame: the norm of the mean residual is likely dominated by a few massive-activation dimensions. All I can conclude is that this configuration hurt.

**A methodological lesson.** My first loop direction was built from Swedish *and* English answers. Only the Swedish answers looped, so the direction partly encoded "Swedish". Steering away from it cut dev loops from 73% to 33%, by moving 93% of Swedish answers out of Swedish (Table D2). On loop rate alone, that would have looked like the best method. **Every "fix" needs its target metric and its regression checks reported in the same run.** This matters for evaluating edits of the article's kind, where the editor may optimise a proxy.

## 6. Process log and deviations

The order of decisions, so readers can judge how much was adaptive:

1. The prompt split, the seeds and the logit-lens edit's settings were fixed before any model output.
2. A 10-prompt sanity check, 6 of whose 9 Swedish prompts and whose 1 English prompt fall in the test split, showed coherent-looking language choice and degenerate content. Its outputs were regenerated deterministically and saved in `results/extra/`. It led to raising generation length from 64 to 96 tokens. It also informed the pivot, which was then confirmed on dev data.
3. Dev probe, then the pivot to looping. The search grids were written after the dev probe.
4. The first search used mixed-language directions, which were confounded. It was rerun with Swedish-only directions; both are kept, and `--mixed-lang` reproduces the first. The steering search was added after the column edit showed little effect on dev.
5. Test run with all arms, then blind grading: Sheets A and B, then Sheet C, added once sampling turned out to remove most loops.
6. **Review of the first draft**, then these arms added *post hoc*:
   - random controls with matched update norm;
   - the s = 1 edit;
   - two more sampling seeds;
   - re-grading of all arms by a single judge (Sheet A2);
   - the 256-token cross-lingual comparison (Sheet B2).

   No method was re-selected after the test results.
7. **Second review round**, then, *post hoc*:
   - 12 more size-matched random edits (15 in total);
   - the pairs regenerated with sampling (Sheet B3);
   - the 10-prompt sanity check regenerated and saved.

   Choosing random seed 2 for grading in Sheet A2 was also post hoc: it was the random edit that had reduced loops on test.
8. Directions were built from, and configurations selected on, the same 30 dev prompts (in-sample), so dev scores are optimistic. Test scores are not affected.

## 7. Limitations

- **Scope:**
  - one small model, chosen as a cheap test case, which turned out to have almost no Swedish competence;
  - greedy decoding;
  - three edit families;
  - a crude loop metric;
  - a 30-prompt dev set.

  The conclusion is scoped to models like this one.
- **Not a test of GPT-6 Astra.** I tested the article's mechanism with fixed procedures. A frontier editor might find a better loop edit. But the decoding arms remove loops almost entirely and still gain nothing in quality, which bounds what any loop fix could achieve here.
- **Quality was graded by blind LLM judges, not native speakers.**
  - Agreement was 94–100% between rounds, and a spot check of 10 items agreed (`results/judging/spot_check.md`). But most items sit at fluency 1 and adequacy 0, which inflates agreement and limits the resolution between arms.
  - The English ≫ Swedish ordering (1.12 against 0.06) is large enough that I don't expect native graders to reverse it. Finer comparisons between arms need native evaluation.
- **Data:** the prompts are synthetic and no native speaker has checked them. One test prompt is even ungrammatical ("Vad är en svart hål?" should be "ett svart hål"). Europarl NLL covers parliamentary register only.
- **No fine-tuning arm.** Steering is the conventional comparison. I left out a LoRA arm because without a licensed Swedish instruction set, and with a model this weak, it would mainly measure the training data. It belongs in the next experiment, on a competent model.

## 8. Recommendation and next steps

**Should SE.LLMA investigate this further?**

- **Not as a route to Swedish quality in models without Swedish competence.** In the model I tested, the Swedish deficit is competence, and data plus continued pre-training (SE.LLMA's plan) is the right tool. This is consistent with the article's own Gemma result.

**Worth adopting:**

1. **Obstacle-versus-competence triage** for any candidate base model or checkpoint:
   - remove obvious obstacles by decoding;
   - measure judged quality on paired Swedish and English questions;
   - only consider an edit where removing the obstacle closes most of the gap.

   For this 0.5B model, it took about an hour of CPU time plus grading; the cost scales with model size.
2. **Sparse edits as audit tools only,** always compared against a *distribution* of size-matched random edits and against decoding fixes. Against three random edits matched only in weights touched, my edit looked effective; against 15 matched in update norm, it is suggestive at best (p = 0.125).

**Most informative next experiment.** Repeat the protocol on a model *with* Swedish competence (a 7B open model, or an early SE.LLMA checkpoint), targeting a plausible obstacle such as Danish/Norwegian leakage. Here it reached 5 of 80 answers at baseline and 8–9 under steering, the repetition penalty and the logit-lens edit. Random edits reached up to 7, so the increase may be noise. Compare at equal budget:

- an LLM-proposed sparse edit, as in the article, with a logged feedback budget;
- a small LoRA on Swedish data;
- decoding and prompting fixes;
- 20+ size-matched random edits, matched on functional change.

Use native-speaker evaluation and a Swedish benchmark (ScandEval/EuroEval).

## 9. Reproducibility

The code, all generations, the grading sheets, grades and key, the judge prompts and the spot check are in the repository. `scripts/reproduce.sh` reruns everything in about 5.5 hours on 2 CPU cores, of which `src/extra_controls.py` takes about 1.5 hours.

- **Seeds:** 1234 for the split and sampling (1234 + 1000k for extra sampling seeds); k = 0–14 for the random directions, with generator seed 1000·k + layer.
- **Package versions:** in `requirements.txt` (torch 2.15.0, transformers 5.23.0).
- **Settings:** `src/config.py`, with its decision order.
- **Dev logs:** record every configuration evaluated.
- **Grading:** LLM-based grading isn't deterministic. To re-grade, fill the `grades_*.json` files from the sheets and run `analyse.py`. `analyse.py` refuses to run if any graded item no longer matches its generation.

## Appendix


**Table D1. Dev search (final, Swedish-only directions)** (dev: 30 Swedish / 15 English prompts; baseline Swedish loop 0.73, Swedish-answer rate 0.97)

| config | sv loop | sv answered in Swedish | en loop | en answered in English |
|---|---|---|---|---|
| loopedit_L6_s1.0 | 0.77 | 0.97 | 0.00 | 1.00 |
| loopedit_L6_s3.0 | 0.67 | 0.97 | 0.13 | 1.00 |
| loopedit_L12_s1.0 | 0.70 | 0.97 | 0.20 | 1.00 |
| loopedit_L12_s3.0 | 0.70 | 0.97 | 0.13 | 1.00 |
| loopedit_L15_s1.0 | 0.60 | 0.97 | 0.07 | 1.00 |
| loopedit_L15_s3.0 | 0.57 | 0.97 | 0.07 | 1.00 |
| loopedit_L18_s1.0 | 0.60 | 0.93 | 0.00 | 1.00 |
| loopedit_L18_s3.0 | 0.60 | 0.37 | 0.13 | 1.00 |
| loopedit_L21_s1.0 | 0.70 | 0.93 | 0.00 | 1.00 |
| loopedit_L21_s3.0 | 0.90 | 0.47 | 0.00 | 1.00 |
| loopedit_L15+19_s1.0 | 0.80 | 0.97 | 0.07 | 1.00 |
| loopedit_L15+19_s3.0 | 0.77 | 0.90 | 0.27 | 1.00 |
| reppen_1.1 | 0.23 | 0.90 | 0.07 | 1.00 |
| reppen_1.2 | 0.00 | 0.93 | 0.07 | 1.00 |
| reppen_1.3 | 0.00 | 0.93 | 0.00 | 1.00 |
| steer_L6_b1.0 | 0.67 | 0.90 | 0.20 | 1.00 |
| steer_L6_b2.0 | 0.77 | 0.83 | 0.53 | 1.00 |
| steer_L12_b1.0 | 0.60 | 0.97 | 0.20 | 1.00 |
| steer_L12_b2.0 | 0.60 | 0.90 | 0.67 | 1.00 |
| steer_L18_b1.0 | 0.53 | 0.97 | 0.13 | 1.00 |
| steer_L18_b2.0 | 0.53 | 0.93 | 0.20 | 1.00 |

Selected by the rule in `src/config.py`: `loopedit_L15_s3.0`, `reppen_1.2`, steering `steer_L18_b1.0`.

**Table D2. First dev search (confounded: directions built from Swedish + English answers)** (dev: 30 Swedish / 15 English prompts; baseline Swedish loop 0.73, Swedish-answer rate 0.97)

| config | sv loop | sv answered in Swedish | en loop | en answered in English |
|---|---|---|---|---|
| loopedit_L6_s1.0 | 0.77 | 0.97 | 0.00 | 1.00 |
| loopedit_L6_s3.0 | 0.63 | 0.97 | 0.00 | 1.00 |
| loopedit_L12_s1.0 | 0.70 | 0.93 | 0.13 | 1.00 |
| loopedit_L12_s3.0 | 0.73 | 0.97 | 0.13 | 1.00 |
| loopedit_L15_s1.0 | 0.67 | 0.97 | 0.07 | 1.00 |
| loopedit_L15_s3.0 | 0.70 | 0.97 | 0.00 | 1.00 |
| loopedit_L18_s1.0 | 0.83 | 0.97 | 0.07 | 1.00 |
| loopedit_L18_s3.0 | 0.87 | 0.90 | 0.33 | 1.00 |
| loopedit_L21_s1.0 | 0.73 | 0.97 | 0.00 | 1.00 |
| loopedit_L21_s3.0 | 0.57 | 0.90 | 0.07 | 1.00 |
| loopedit_L15+19_s1.0 | 0.63 | 0.90 | 0.13 | 1.00 |
| loopedit_L15+19_s3.0 | 0.37 | 0.80 | 0.00 | 1.00 |
| reppen_1.1 | 0.23 | 0.90 | 0.07 | 1.00 |
| reppen_1.2 | 0.00 | 0.93 | 0.07 | 1.00 |
| reppen_1.3 | 0.00 | 0.93 | 0.00 | 1.00 |
| steer_L6_b1.0 | 0.33 | 0.07 | 0.27 | 1.00 |
| steer_L6_b2.0 | 0.57 | 0.03 | 0.60 | 1.00 |

Selected by the rule in `src/config.py`: `loopedit_L21_s3.0`, `reppen_1.2`.

### AI assistance
I carried out this work with substantial help from an AI assistant (Claude), which wrote most of the code and drafted text under my direction. Answer quality was graded blind by LLM judges. Details are in `AI_USAGE.md` in the repository.
