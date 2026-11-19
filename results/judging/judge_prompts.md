# Judge prompts (verbatim)

Every judge was a fresh Claude Opus 5.5 subagent instance (general-purpose agent type). None had seen the experiment, its code, its hypothesis or the arm names.

| Judge | Sheet | Content |
|---|---|---|
| 1 | A | 6 arms × 40 Swedish test prompts |
| 2 | B | 50 sv/en pairs, 96 tokens |
| 3 | C | sampling arm + 40 re-graded A items |
| 4 | A2 | 9 arms × 40, main quality result |
| 5 | B2 | 50 pairs, 256 tokens, greedy |
| 6 | B3 | 50 pairs, 256 tokens, sampling |

The mapping from items to arms (`KEY_do_not_show_judges.json`) was applied afterwards by `src/analyse.py`.

## Template for Swedish-only sheets (A, C, A2)
`{N}` = number of items, `{FILE}` = sheet path, `{OUT}` = grades path, `{PFX}` = item-id prefix. `[A2 only: …]` marks text present only in the A2 prompt.

```
You are grading chatbot answers for a language-quality study. Read the JSON file {FILE}. It is a list of {N} items, each {item_id, prompt, response}. The prompts are in Swedish. Do NOT open any other files in /home/claude/sellma-wsp (grade only from this sheet), and do not try to infer where answers came from.[A2 only:  Many answers may look similar; grade each on its own text.]

Grade EVERY item on two 0-2 scales, judging each item on its own merits:
- fluency (in the language of the prompt, i.e. Swedish): 0 = not usable Swedish (garbled, badly ungrammatical, invented words, or mostly in another language such as English/Norwegian/Danish); 1 = understandable Swedish but with clear grammatical or word errors; 2 = grammatical, natural Swedish.
- adequacy: 0 = wrong, irrelevant, nonsensical, empty, a refusal, or pure repetition; 1 = partly relevant/correct but with factual errors, confusions or major omissions; 2 = correct, relevant and helpful.
Responses were cut off at ~96 tokens: do not penalise an answer for ending mid-sentence. Do penalise repetition loops (repeating the same phrase) under adequacy and, if it breaks the language, fluency. Be strict and consistent; a fluent but factually wrong answer gets fluency 2, adequacy 0.

Work through all items carefully (you may read the file in chunks). Write your grades as a JSON list to {OUT} with exactly this form: [{"item_id": "{PFX}000", "fluency": 0, "adequacy": 0}, ...] containing all {N} item_ids, integers only. You may write a small Python helper to assemble the JSON, but the grades themselves must be your own judgments of each text, not computed heuristics. When finished, verify the file has {N} entries, and reply with a short note on how hard grading was and any systematic patterns you saw in the answers (without guessing at their origin).
```
(Closing sentences varied slightly between sheets: "Read every item fully", "reply with a 2-sentence note".)

## Template for bilingual sheets (B, B2, B3)
`{T}` = 96 for B, 256 for B2/B3.

```
You are grading chatbot answers for a language-quality study. Read the JSON file {FILE}. It is a list of 100 items, each {item_id, prompt, response}. Prompts are in Swedish or in English. Do NOT open any other files in /home/claude/sellma-wsp (grade only from this sheet), and do not try to infer where answers came from.

Grade EVERY item on two 0-2 scales, judging each item on its own merits:
- fluency (in the language of the PROMPT): 0 = not usable in that language (garbled, badly ungrammatical, invented words, or mostly in a different language than the prompt); 1 = understandable but with clear grammatical or word errors; 2 = grammatical, natural.
- adequacy: 0 = wrong, irrelevant, nonsensical, empty, a refusal, or pure repetition; 1 = partly relevant/correct but with factual errors, confusions or major omissions; 2 = correct, relevant and helpful.
Responses were cut off at ~{T} tokens: do not penalise an answer for ending mid-sentence. Do penalise repetition loops under adequacy[B2/B3 only: (an answer that is correct at first and then degenerates into repeating itself should lose at least one adequacy point)]. Be strict and use the same standard for both languages; a fluent but factually wrong answer gets fluency 2, adequacy 0.

Write your grades as a JSON list to {OUT} with exactly this form: [{"item_id": "{PFX}000", "fluency": 0, "adequacy": 0}, ...] containing all 100 item_ids, integers only. You may write a small Python helper to assemble the JSON, but the grades themselves must be your own judgments of each text, not computed heuristics. When finished, verify the file has 100 entries, and reply with a short note.
```
