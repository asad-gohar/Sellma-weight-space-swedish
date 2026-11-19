"""Review round 2 (post hoc): more evidence for two comparisons.

1. A null distribution for the targeted edit: 12 more random directions (seeds 3-14),
   each rescaled to the loop edit's update norm, on the 80 Swedish test prompts.
   With seeds 0-2 from run_test.py that gives 15 size-matched random edits.
2. The 10-prompt sanity check that preceded the dev probe (64 tokens, greedy),
   regenerated and saved so the process log is auditable.
3. The 50 Swedish/English pairs regenerated at 256 tokens with sampling
   (T=0.7, top-p 0.8, repetition penalty 1.05, top-k off, seed 1234). This does not
   remove loops at 256 tokens, but it equalises them across languages (sv 0.46, en 0.44),
   so a remaining gap cannot be due to Swedish looping more.
"""
import json
import re
import sys

import torch

import config as C
from arms import load_split, run_split
from build_pairs import PAIRS
from common import ROOT, generate, load_model, score_response, write_jsonl
from dev_search import Multi
from edits import ColumnProjectionEdit, random_direction

OUT = ROOT / "results" / "extra"
tok, model = load_model()
log = json.load(open(ROOT / "results" / "dev" / "dev_search_log.json"))
_, Ls, s = log["selected_edit"].split("_")
layers, s = [int(x) for x in Ls[1:].split("+")], float(s[1:])
loop_dirs = {L: torch.load(ROOT / "results" / "dev" / f"loop_dir_L{L}.pt") for L in layers}
frob = {L: ColumnProjectionEdit(model, L, loop_dirs[L], 64, s).frob for L in layers}

if "sanity" in sys.argv or len(sys.argv) == 1:
    prompts = ["Varför är himlen blå?", "Förklara vad ett API är.", "Why is the sky blue?",
               "Skriv en kort dikt om hösten.", "Vad är allemansrätten?", "Hur fungerar kryptering?",
               "Ge mig tre tips för att sova bättre.", "Vad är en Docker-container?",
               "Hur kokar man perfekta ägg?", "Vad är 15 procent av 240?"]
    test_p = {r["prompt"] for r in load_split("sv_test")}
    outs = generate(tok, model, prompts, max_new_tokens=64, batch_size=10)
    write_jsonl(OUT / "sanity_check_64tok.jsonl",
                [{"prompt": p, "in_test_split": p in test_p, "response": o} for p, o in zip(prompts, outs)])
    print("sanity saved", flush=True)

if "pairs" in sys.argv or len(sys.argv) == 1:
    rows = []
    for lang, idx in (("sv", 0), ("en", 1)):
        ps = [p[idx] for p in PAIRS]
        outs = generate(tok, model, ps, max_new_tokens=256, batch_size=10, do_sample=True, temperature=0.7,
                        top_p=0.8, repetition_penalty=1.05, seed=C.SEED)
        for k, (p, o) in enumerate(zip(ps, outs)):
            rows.append({"pair": k, "lang": lang, "prompt": p, "response": o,
                         "n_words": len(re.findall(r"\w+", o)), **score_response(o, lang)})
    write_jsonl(ROOT / "results" / "crosslingual" / "pairs_256tok_sampling.jsonl", rows)
    for lang in ("sv", "en"):
        r = [x for x in rows if x["lang"] == lang]
        print("pairs-sampling", lang, sum(x["loop"] for x in r) / 50, sum(x["n_words"] for x in r) / 50, flush=True)

if "random" in sys.argv or len(sys.argv) == 1:
    sv = load_split("sv_test")
    summ_p = OUT / "random_normmatched_summary.json"
    summ = json.load(open(summ_p)) if summ_p.exists() else {}
    for k in range(3, 15):
        if str(k) in summ:
            continue
        dirs = {L: random_direction(model.config.hidden_size, 1000 * k + L) for L in layers}
        ctx = Multi(*[ColumnProjectionEdit(model, L, dirs[L], 64, s, target_frob=frob[L]) for L in layers])
        res, _ = run_split(tok, model, sv, ctx=ctx)
        write_jsonl(OUT / f"random_edit_normmatched_seed{k}_sv.jsonl", res)
        summ[str(k)] = {"loop": sum(r["loop"] for r in res) / len(res),
                        "on_target": sum(r["on_target"] for r in res) / len(res)}
        json.dump(summ, open(summ_p, "w"), indent=1)
        print("random seed", k, summ[str(k)], flush=True)
