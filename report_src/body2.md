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
