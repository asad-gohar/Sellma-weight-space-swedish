# Held-out test results

Loop edit selected on dev: `loopedit_L15_s3.0`, 57,344 weights; update Frobenius norm 3.27. Unmatched random updates had norm 1.28, 1.36, 1.31.

## 1. Automatic metrics (80 Swedish, 35 English test prompts; greedy unless stated)

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

## 2. Targeted edit vs random edits (paired, 80 Swedish prompts; Δ loop rate)

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

## 3. Blind quality grading, one judge for all arms (Sheet A2: 40 Swedish test prompts per arm)

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

## 3b. First grading round (Sheet A: 6 arms, judge 1; Sheet C: sampling arm, judge 3)

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

## 4. Cross-lingual gap: the same 50 questions in English and Swedish, baseline model, 256-token budget (Sheet B2, judge 5)

Answers at 256 new tokens: mean 119 words in Swedish and 164 in English. Loop rate: Swedish 0.86, English 0.72.

- Mean adequacy: English 1.12, Swedish 0.06; difference +1.06 [+0.88, +1.24] (n=50 pairs)
- Mean fluency: English 2.00, Swedish 0.72
- 1 of the 14 questions answered fully adequately in English were also adequate in Swedish

## 4a. Cross-lingual gap with sampling, loop rates equal across languages (Sheet B3, judge 6)

Sampling (T=0.7, top-p 0.8, repetition penalty 1.05, top-k off), 256 tokens. Loop rate: Swedish 0.46, English 0.44; mean words 109 / 176.

- Mean adequacy: English 1.24, Swedish 0.10; difference +1.14 [+0.92, +1.36] (n=50 pairs)
- Mean fluency: English 2.00, Swedish 0.32
- 2 of the 21 questions answered fully adequately in English were also adequate in Swedish

## 4b. First version (Sheet B, judge 2, 96-token budget; Swedish answers about a third shorter)


- Mean adequacy: English 1.32, Swedish 0.08; difference +1.24 [+1.04, +1.44] (n=50 pairs)
- Mean fluency: English 2.00, Swedish 0.80
- 1 of the 23 questions answered fully adequately in English were also adequate in Swedish
