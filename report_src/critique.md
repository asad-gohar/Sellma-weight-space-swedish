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
