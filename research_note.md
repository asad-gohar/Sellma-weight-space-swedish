# Is Swedish degeneration in a small open model a removable obstacle?
*Asad Gohar, November 2026*

*A weight-space-programming check for SE.LLMA. Research note; methods, all tables and the process log are in `report.pdf`.*

## 1. The article, critically
*Sparks of Weight-Space Programming* mixes two regimes.

- **Reasoning about weights with no forward passes** is the strongest evidence: copying heads found in transformers Astra couldn't run, and MinAtar policies written one-shot. All are small networks the authors built.
- **Edits found with model feedback** is where the headline result sits. Qwen3.5-0.8B's MATH-500 reaches 48.1 only at 50 passes (12–14 at 10 and 25), in single runs with no random-edit or fine-tuning control. The edit mainly removes loops (93% → 33% of answers hit the token limit, despite a presence penalty in both arms). Nothing helped Gemma, which rarely loops, and reading a real LM's behaviour from its weights alone failed.

**Justified:** a sparse edit *found with feedback* can remove an obstacle that hides an existing ability.
**Not yet justified:** that LLMs read *LLM* weights natively, or that edits beat fine-tuning.

## 2. Question and hypothesis
SE.LLMA needs to know whether a weak Swedish model has a **removable obstacle** or is **missing competence**. **H:** Swedish failures in Qwen2.5-0.5B-Instruct are largely a removable obstacle. It is supported if a targeted sparse edit reduces the failure more than size-matched random edits, without regressions, and quality then rises toward the model's English level. It is challenged if neither holds.

**How the target changed.** I first targeted wrong-language answers, but on dev data the model answered in Swedish 97% of the time while 73% of its Swedish answers looped (0% of English ones at 96 tokens). So I switched to looping, which is also the article's failure mode. An earlier 10-prompt sanity check (7 of its 10 prompts are test prompts; saved in `results/extra/`) also informed this switch and the 96-token length.

## 3. Experiment
Qwen2.5-0.5B-Instruct (Ollama fp16, sha256 `6f96e01a…`), CPU, greedy decoding, 96 new tokens. 110 Swedish and 50 English prompts written for the study, split with a fixed seed into dev (30/15) and test (80/35); all selection used dev only. Metrics: answer language, loop rate (a word 5-gram repeats), ΔNLL on held-out Europarl, and blind 0–2 fluency and adequacy grades from one fresh LLM judge for all arms.

**Arms.** The **loop edit** (article-style) removes the direction separating looping from non-looping Swedish MLP outputs from the 64 `down_proj` columns that write most along it (57,344 weights). Dev selected layer 15 at s = 3, an overshoot that flips the component to −2×; s = 1, the literal removal, is also reported. Controls: **15 random edits** rescaled to the same update norm, **activation steering**, a no-feedback **logit-lens edit**, **decoding** (repetition penalty 1.2; sampling at T 0.7, top-p 0.8, repetition penalty 1.05, top-k off, 3 seeds), a **Swedish system prompt**, and the unchanged **baseline**.

## 4. Results (held-out test, 80 Swedish prompts)
| Arm | Loop rate | Δ loop vs baseline [95% CI] | In Swedish | Adequacy* | Δ fluency* |
|---|---|---|---|---|---|
| Baseline | 0.78 | – | 0.93 | 0.12 | – |
| Loop edit, s=3 | 0.64 | −0.14 [−0.25, −0.03] | 0.93 | 0.15 | 0.00 |
| Loop edit, s=1 | 0.71 | −0.06 [−0.16, +0.04] | 0.91 | 0.15 | +0.03 |
| Random, size-matched (15) | 0.62–0.78 (median 0.72) | – | 0.88–0.94 | 0.15, 0.07 | ≈0 |
| Steering | 0.62 | −0.15 [−0.29, −0.01] | 0.86 | 0.10 | −0.12 [−0.25, 0.00] |
| Logit-lens edit | 0.84 | +0.06 [−0.04, +0.16] | 0.84 | – | – (ΔNLL +0.57) |
| Repetition penalty 1.2 | 0.03 | −0.75 [−0.84, −0.66] | 0.89 | 0.07 | −0.17 [−0.30, −0.05] |
| Sampling (3 seeds) | 0.07–0.14 | −0.64 to −0.70 | 0.89–0.91 | 0.10 | −0.07 [−0.23, +0.07] |
| Swedish system prompt | 0.78 | 0.00 | 0.96 | 0.10 | +0.15 [+0.03, +0.28] |

\*Graded on 40 prompts, one judge for all arms. Sampling: seed 0 only. Random: 2 of the seeds. Δ is paired against baseline.

**Quality.** No arm produced a usable Swedish answer (adequacy 2 on 0 of 40 prompts in every arm), and no arm's adequacy differs from baseline by more than 0.05. Answers that loop score lower than those that don't (0.08 against 0.15), partly by construction, since the rubric tells judges to penalise loops. But none of 155 loop-free answers is fully adequate. **On the same 50 questions in English** (baseline, 256 tokens), adequacy is **1.12 against 0.06** with greedy decoding. With sampling, which equalises looping across languages (44% English, 46% Swedish), it is **1.24 against 0.10**, difference +1.14 [+0.92, +1.36].

**Regressions.** The Swedish system prompt makes 31% of answers to *English* prompts come out non-English. Danish/Norwegian leakage rises from 5 to 8–9 of 80 answers under steering, the logit-lens edit and the repetition penalty. Random edits reach up to 7 of 80, so this may be noise. **A confound I caught:** a loop direction built partly from English answers "fixed" dev loops (73% → 33%) by moving 93% of answers out of Swedish.

## 5. Interpretation
1. **The article-style edit's effect is not established.** It beats baseline only at the overshooting strength. Against 15 random edits rescaled to the same update norm (loop rates 0.62–0.78), it beat 14 and one did slightly better: empirical p = 0.125, i.e. (1+1)/(15+1). Random edits on their own cut loops by about 6 points, about 40% of the edit's effect. The literal removal (s = 1) has no significant effect. Even equal update norm isn't equal functional size: on the 80 Swedish prompts, random edits change 44–56 outputs, the edit 70. Steering does about as well as the edit, at a fluency cost.
2. **Decoding removes most loops**, sampling with no significant fluency cost. In the article's long reasoning chains a presence penalty wasn't enough, so the need for an edit depends on the setting.
3. **The deficit is competence, not a hidden ability.** With loops gone, Swedish adequacy stays near 0.1/2, while the same questions in English score above 1. This is the article's Gemma case.
4. **The no-feedback logit-lens edit hurt** in its one configuration (ΔNLL +0.57). It is a heuristic, not an LLM reading weights.

**Limitations.** One 0.5B model with near-zero Swedish competence, so the competence finding was likely from the start. A crude loop metric. Synthetic prompts no native speaker has checked (one is ungrammatical: "en svart hål"). LLM judges at a floor effect, which inflates their 94–100% agreement. Directions built and selected on the same 30 dev prompts. The article's mechanism was tested with fixed procedures, not a frontier editor. Several arms were added after two review rounds; all are marked *post hoc* in the report's process log.

## 6. Recommendation
**For models without Swedish competence, weight editing is not a route to better Swedish.** Data and continued pre-training, which is SE.LLMA's plan, are the tools for that. My evidence comes from one small model, so whether edits help *competent* Swedish models remains open.

**Worth adopting:** (1) an **obstacle-versus-competence triage**: remove the obvious obstacle by decoding, compare judged quality on paired Swedish and English questions, and only consider an edit where removing the obstacle closes most of the gap. For this 0.5B model it took about an hour of CPU time plus grading. (2) **Sparse edits only as audit tools**, always tested against a *distribution* of size-matched random edits. Against three random edits matched only in weights touched, my edit looked effective; against 15 matched in update norm, it is suggestive at best (p = 0.125).

**Next experiment:** the same protocol on a model *with* Swedish competence (7B, or an SE.LLMA checkpoint), targeting Danish/Norwegian leakage. Compare at equal budget an LLM-proposed sparse edit, a small Swedish LoRA, decoding fixes and 20+ size-matched random edits, with native-speaker evaluation and ScandEval/EuroEval.
