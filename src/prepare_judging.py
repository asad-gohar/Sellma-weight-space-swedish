"""Step 4: build BLIND grading sheets (arm names hidden, order shuffled).

Sheet A: first 40 Swedish TEST prompts x 6 arms (quality of Swedish answers).
Sheet C: the arm with Qwen's recommended sampling (same 40 prompts) plus 40 sheet-A
         items re-graded by a fresh judge (inter-judge agreement). Added after
         sheets A/B were graded, once the sampling arm turned out to remove most loops.
Sheet B: the 50 Swedish/English prompt pairs, baseline answers only (cross-lingual
         gap: does the model know the answer in English but not in Swedish?).
Sheet A2: one fresh judge grades all arms on the same 40 prompts (main quality result).
Sheet B2: the 50 pairs answered at 256 tokens, greedy (main cross-lingual result).
Sheet B3: the 50 pairs answered at 256 tokens with sampling (loop rates equal across languages).
The key mapping item_id -> arm is written to a separate file the judges never see.
Generation is deterministic on a fixed setup; analyse.py checks that every graded item
still matches its generation, so stale grades can never attach to new text.
"""
import json
import random

from common import ROOT, read_jsonl
from build_pairs import PAIRS

OUT = ROOT / "results" / "judging"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
summary = json.load(open(ROOT / "results" / "test" / "summary.json"))
names = [k for k in summary if k != "meta"]
pick = {
    "baseline": "baseline",
    "reppen": next(n for n in names if n.startswith("repetition_penalty")),
    "loop_edit": next(n for n in names if n.startswith("loop_edit")),
    "random_edit": "random_edit_seed0",
    "sv_system_prompt": "sv_system_prompt",
}
steer = next((n for n in names if n.startswith("steering")), None)
if steer:
    pick["steering"] = steer

items, key = [], {}
for short, arm in pick.items():
    rows = [r for r in read_jsonl(ROOT / "results" / "test" / f"{arm}.jsonl") if r["lang"] == "sv"][:40]
    for r in rows:
        items.append({"prompt": r["prompt"], "response": r["response"], "_arm": short, "_pid": r["id"]})
rng.shuffle(items)
sheetA = []
for i, it in enumerate(items):
    iid = f"A{i:03d}"
    key[iid] = {"arm": it["_arm"], "prompt_id": it["_pid"]}
    sheetA.append({"item_id": iid, "prompt": it["prompt"], "response": it["response"]})

# Sheet B: baseline answers for paired prompts (dev + test files, both languages)
resp = {}
for f in ["dev/sv_dev__baseline", "dev/en_dev__baseline"]:
    for r in read_jsonl(ROOT / "results" / f"{f}.jsonl"):
        resp[r["prompt"]] = r["response"]
for r in read_jsonl(ROOT / "results" / "test" / "baseline.jsonl"):
    resp[r["prompt"]] = r["response"]
itemsB = []
for k, (sv, en) in enumerate(PAIRS):
    itemsB.append({"prompt": sv, "response": resp[sv], "_lang": "sv", "_pair": k})
    itemsB.append({"prompt": en, "response": resp[en], "_lang": "en", "_pair": k})
rng.shuffle(itemsB)
sheetB = []
for i, it in enumerate(itemsB):
    iid = f"B{i:03d}"
    key[iid] = {"lang": it["_lang"], "pair": it["_pair"]}
    sheetB.append({"item_id": iid, "prompt": it["prompt"], "response": it["response"]})

json.dump(sheetA, open(OUT / "sheet_A.json", "w"), indent=1, ensure_ascii=False)
json.dump(sheetB, open(OUT / "sheet_B.json", "w"), indent=1, ensure_ascii=False)
json.dump({"arms": pick, "key": key}, open(OUT / "KEY_do_not_show_judges.json", "w"), indent=1)

# Sheet C
r = random.Random(11)
K = json.load(open(OUT / "KEY_do_not_show_judges.json"))
samp = [x for x in read_jsonl(ROOT / "results" / "test" / "qwen_recommended_sampling.jsonl") if x["lang"] == "sv"][:40]
rep = r.sample(sheetA, 40)
itemsC = ([{"prompt": s["prompt"], "response": s["response"], "_k": {"arm": "qwen_sampling", "prompt_id": s["id"]}} for s in samp]
          + [{"prompt": a["prompt"], "response": a["response"], "_k": {"repeat_of": a["item_id"]}} for a in rep])
r.shuffle(itemsC)
sheetC = []
for i, it in enumerate(itemsC):
    iid = f"C{i:03d}"
    K["key"][iid] = it["_k"]
    sheetC.append({"item_id": iid, "prompt": it["prompt"], "response": it["response"]})
K["arms"]["qwen_sampling"] = "qwen_recommended_sampling"
json.dump(sheetC, open(OUT / "sheet_C.json", "w"), indent=1, ensure_ascii=False)
json.dump(K, open(OUT / "KEY_do_not_show_judges.json", "w"), indent=1)
print(len(sheetA), len(sheetB), len(sheetC))

# ---------------------------------------------------------------- review round
# Sheet A2: ONE fresh judge grades every arm (fixes "different judges for different
# arms"), including the arms added after review. Sheet B2: the cross-lingual pairs
# regenerated with a 256-token budget so Swedish answers are not shorter.
import os
if (ROOT / "results" / "crosslingual" / "pairs_256tok.jsonl").exists():
    summary = json.load(open(ROOT / "results" / "test" / "summary.json"))
    names = [k for k in summary if k != "meta"]
    pickA2 = {
        "baseline": "baseline",
        "loop_edit": next(n for n in names if n.startswith("loop_edit[")),
        "loop_edit_s1": next(n for n in names if n.startswith("loop_edit_s1")),
        "random_edit_normmatched": "random_edit_normmatched_seed0",
        # seed 2 added because it is the random edit that DID reduce loops (0.63):
        # lets us ask whether loop reduction by an untargeted edit changes quality
        "random_edit_normmatched_s2": "random_edit_normmatched_seed2",
        "sv_system_prompt": "sv_system_prompt",
        "reppen": next(n for n in names if n.startswith("repetition_penalty")),
        "steering": next(n for n in names if n.startswith("steering")),
        "qwen_sampling": "qwen_recommended_sampling",
    }
    r2 = random.Random(23)
    items = []
    for short, arm in pickA2.items():
        rows = [x for x in read_jsonl(ROOT / "results" / "test" / f"{arm}.jsonl") if x["lang"] == "sv"][:40]
        items += [{"prompt": x["prompt"], "response": x["response"], "_k": {"arm2": short, "prompt_id": x["id"]}}
                  for x in rows]
    r2.shuffle(items)
    K = json.load(open(OUT / "KEY_do_not_show_judges.json"))
    sheetA2 = []
    for i, it in enumerate(items):
        iid = f"D{i:03d}"
        K["key"][iid] = it["_k"]
        sheetA2.append({"item_id": iid, "prompt": it["prompt"], "response": it["response"]})
    itemsB2 = [{"prompt": x["prompt"], "response": x["response"], "_k": {"lang2": x["lang"], "pair": x["pair"]}}
               for x in read_jsonl(ROOT / "results" / "crosslingual" / "pairs_256tok.jsonl")]
    r2.shuffle(itemsB2)
    sheetB2 = []
    for i, it in enumerate(itemsB2):
        iid = f"E{i:03d}"
        K["key"][iid] = it["_k"]
        sheetB2.append({"item_id": iid, "prompt": it["prompt"], "response": it["response"]})
    K["arms_A2"] = pickA2
    json.dump(sheetA2, open(OUT / "sheet_A2.json", "w"), indent=1, ensure_ascii=False)
    json.dump(sheetB2, open(OUT / "sheet_B2.json", "w"), indent=1, ensure_ascii=False)
    json.dump(K, open(OUT / "KEY_do_not_show_judges.json", "w"), indent=1)
    print("A2", len(sheetA2), "B2", len(sheetB2))

# Sheet B3 (review round 2): the 50 pairs at 256 tokens with sampling, where loop rates
# are equal across languages (sv 0.46, en 0.44; greedy B2: sv 0.86, en 0.72).
p3 = ROOT / "results" / "crosslingual" / "pairs_256tok_sampling.jsonl"
if p3.exists():
    r3 = random.Random(31)
    K = json.load(open(OUT / "KEY_do_not_show_judges.json"))
    items3 = [{"prompt": x["prompt"], "response": x["response"], "_k": {"lang3": x["lang"], "pair": x["pair"]}}
              for x in read_jsonl(p3)]
    r3.shuffle(items3)
    sheetB3 = []
    for i, it in enumerate(items3):
        iid = f"F{i:03d}"
        K["key"][iid] = it["_k"]
        sheetB3.append({"item_id": iid, "prompt": it["prompt"], "response": it["response"]})
    json.dump(sheetB3, open(OUT / "sheet_B3.json", "w"), indent=1, ensure_ascii=False)
    json.dump(K, open(OUT / "KEY_do_not_show_judges.json", "w"), indent=1)
    print("B3", len(sheetB3))
