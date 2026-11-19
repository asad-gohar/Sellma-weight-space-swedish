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
