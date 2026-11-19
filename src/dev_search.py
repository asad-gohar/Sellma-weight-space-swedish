"""Step 2: develop/select methods on DEV prompts only (repeated model feedback).

Builds the article-style loop direction from the model's own baseline DEV
generations, then searches layer x strength on dev. Also tunes the inference-time
alternative (repetition penalty) with the same dev prompts, so both get feedback.
Every configuration evaluated is counted and logged (= the 'feedback budget').
"""
import json
import time

import torch

import config as C
from arms import load_split, run_split, summarise
from common import ROOT, build_chat, load_model, read_jsonl, write_jsonl, DEFAULT_SYSTEM
from edits import ColumnProjectionEdit, loop_direction

import sys

DEV = ROOT / "results" / "dev"          # dev baseline generations live here
# --mixed-lang reproduces the FIRST (confounded) search: directions built from
# Swedish + English answers; outputs go to results/dev_v1_confounded/.
MIXED = "--mixed-lang" in sys.argv
OUT = ROOT / "results" / ("dev_v1_confounded" if MIXED else "dev")
LAYERS = [6, 12, 15, 18, 21]
STRENGTHS = [1.0, 3.0]
REP_PENALTIES = [1.1, 1.2, 1.3]


class Multi:
    def __init__(self, *ctxs):
        self.ctxs = ctxs

    def __enter__(self):
        for c in self.ctxs:
            c.__enter__()
        return self

    def __exit__(self, *a):
        for c in reversed(self.ctxs):
            c.__exit__(*a)


def main():
    torch.manual_seed(C.SEED)
    tok, model = load_model()
    sv_dev, en_dev = load_split("sv_dev"), load_split("en_dev")
    # v2: directions from SWEDISH dev answers only. v1 mixed in English answers, so
    # "looping vs not" was confounded with "Swedish vs English" (English dev answers didn't loop at 96 tokens)
    # and subtracting the direction switched the model to English. See
    # results/dev_v1_confounded/.
    base = read_jsonl(DEV / "sv_dev__baseline.jsonl")
    if MIXED:
        base += read_jsonl(DEV / "en_dev__baseline.jsonl")
    chats = [build_chat(tok, r["prompt"], DEFAULT_SYSTEM) for r in base]
    resps = [r["response"] for r in base]

    base_sv = summarise(read_jsonl(DEV / "sv_dev__baseline.jsonl"))
    base_en = summarise(read_jsonl(DEV / "en_dev__baseline.jsonl"))
    log = {"baseline": {"sv": base_sv, "en": base_en}, "configs": []}

    dirs = {}
    for L in sorted(set(LAYERS) | {15, 19}):
        d, counts = loop_direction(model, tok, chats, resps, L)
        dirs[L] = d
        torch.save(d, OUT / f"loop_dir_L{L}.pt")
    log["loop_token_counts"] = counts

    def evaluate(name, ctx=None, **kw):
        t0 = time.time()
        rs, _ = run_split(tok, model, sv_dev, ctx=ctx, **kw)
        re_, _ = run_split(tok, model, en_dev, ctx=ctx, **kw)
        entry = {"name": name, "sv": summarise(rs), "en": summarise(re_),
                 "seconds": round(time.time() - t0, 1)}
        log["configs"].append(entry)
        write_jsonl(OUT / "search" / f"{name}.jsonl", rs + re_)
        print(name, "sv loop %.2f on %.2f | en loop %.2f on %.2f" % (
            entry["sv"]["loop"], entry["sv"]["on_target"], entry["en"]["loop"], entry["en"]["on_target"]),
            flush=True)
        json.dump(log, open(OUT / "dev_search_log.json", "w"), indent=1, ensure_ascii=False)

    for L in LAYERS:
        for s in STRENGTHS:
            evaluate(f"loopedit_L{L}_s{s}", ColumnProjectionEdit(model, L, dirs[L], 64, s))
    # article-like two-layer variant (the post edited MLP layers 15 and 19)
    for s in STRENGTHS:
        evaluate(f"loopedit_L15+19_s{s}",
                 Multi(ColumnProjectionEdit(model, 15, dirs[15], 64, s),
                       ColumnProjectionEdit(model, 19, dirs[19], 64, s)))
    for rp in REP_PENALTIES:
        evaluate(f"reppen_{rp}", repetition_penalty=rp)

    # selection rule (fixed in this file before the search ran): lowest sv loop rate s.t. sv on-target >= 0.9 and
    # en on-target >= baseline_en - tolerance; ties -> single-layer before two-layer, then alphabetical name
    def ok(e):
        return e["sv"]["on_target"] >= 0.9 and e["en"]["on_target"] >= base_en["on_target"] - C.EN_TOLERANCE
    edits = [e for e in log["configs"] if e["name"].startswith("loopedit") and ok(e)]
    reps = [e for e in log["configs"] if e["name"].startswith("reppen") and ok(e)]
    key = lambda e: (e["sv"]["loop"], "+" in e["name"], e["name"])
    log["selected_edit"] = min(edits, key=key)["name"] if edits else None
    log["selected_reppen"] = min(reps, key=key)["name"] if reps else None
    log["n_dev_evaluations"] = len(log["configs"])
    json.dump(log, open(OUT / "dev_search_log.json", "w"), indent=1, ensure_ascii=False)
    print("selected", log["selected_edit"], log["selected_reppen"])


if __name__ == "__main__":
    main()
