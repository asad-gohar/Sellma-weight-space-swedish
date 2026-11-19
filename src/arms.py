"""Running a prompt set under one 'arm' (baseline / prompt / sampling / edit) and scoring it."""
from __future__ import annotations

import contextlib
import time

from common import DEFAULT_SYSTEM, SV_SYSTEM, generate, read_jsonl, score_response, ROOT
import config as C


def load_split(name):
    return read_jsonl(ROOT / "data" / f"prompts_{name}.jsonl")


def run_split(tok, model, rows, system=DEFAULT_SYSTEM, ctx=None, user_suffix="", **gen_kw):
    ctx = ctx or contextlib.nullcontext()
    prompts = [r["prompt"] + user_suffix for r in rows]
    t0 = time.time()
    with ctx:
        outs = generate(tok, model, prompts, system=system, max_new_tokens=C.MAX_NEW_TOKENS,
                        batch_size=C.GEN_BATCH, **gen_kw)
    dt = time.time() - t0
    res = []
    for r, o in zip(rows, outs):
        s = score_response(o, r["lang"])
        res.append({**r, "response": o, **s})
    return res, dt


def summarise(res):
    n = len(res)
    if n == 0:
        return {}
    labs = {}
    for r in res:
        labs[r["label"]] = labs.get(r["label"], 0) + 1
    return {
        "n": n,
        "on_target": sum(r["on_target"] for r in res) / n,
        "pure": sum(r["pure"] for r in res) / n,
        "sent_frac": sum(r["sent_frac"] for r in res) / n,
        "cjk": sum(r["has_cjk"] for r in res) / n,
        "distinct": sum(r["distinct_word_ratio"] for r in res) / n,
        "rep4": sum(r["rep4"] for r in res) / n,
        "loop": sum(r["loop"] for r in res) / n,
        "labels": labs,
    }
