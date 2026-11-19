"""Step 2b (added after the column edit showed no dev effect): a stronger,
conventional activation-steering baseline that subtracts the loop direction from
the residual stream (all positions), with a principled scale (beta=1 maps the mean
looping state onto the mean non-looping state). Dev prompts only."""
import json
import time

import torch

from arms import load_split, run_split, summarise
from common import ROOT, build_chat, load_model, read_jsonl, write_jsonl, DEFAULT_SYSTEM
from edits import ResidualHook, loop_residual_difference

import sys

DEV = ROOT / "results" / "dev"          # dev baseline generations live here
# --mixed-lang reproduces the FIRST (confounded) search: directions built from
# Swedish + English answers; outputs go to results/dev_v1_confounded/.
MIXED = "--mixed-lang" in sys.argv
OUT = ROOT / "results" / ("dev_v1_confounded" if MIXED else "dev")
LAYERS = [6, 12, 18]
BETAS = [1.0, 2.0]


def main():
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
    diffs, counts = loop_residual_difference(model, tok, chats, [r["response"] for r in base], LAYERS)
    for L, d in diffs.items():
        torch.save(d, OUT / f"loop_resdiff_L{L}.pt")
    log = json.load(open(OUT / "dev_search_log.json"))
    log.setdefault("configs2", [])
    log["resdiff_norms"] = {L: d.norm().item() for L, d in diffs.items()}
    for L in LAYERS:
        for b in BETAS:
            name = f"steer_L{L}_b{b}"
            t0 = time.time()
            ctx = ResidualHook(model, L, -b * diffs[L])
            rs, _ = run_split(tok, model, sv_dev, ctx=ctx)
            re_, _ = run_split(tok, model, en_dev, ctx=ctx)
            e = {"name": name, "sv": summarise(rs), "en": summarise(re_), "seconds": round(time.time() - t0, 1)}
            log["configs2"].append(e)
            write_jsonl(OUT / "search" / f"{name}.jsonl", rs + re_)
            json.dump(log, open(OUT / "dev_search_log.json", "w"), indent=1, ensure_ascii=False)
            print(name, "sv loop %.2f on %.2f | en loop %.2f on %.2f" % (
                e["sv"]["loop"], e["sv"]["on_target"], e["en"]["loop"], e["en"]["on_target"]), flush=True)


if __name__ == "__main__":
    main()
