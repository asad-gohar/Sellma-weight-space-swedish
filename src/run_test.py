"""Step 3: held-out TEST evaluation of every arm, plus Europarl NLL regressions.

Nothing here is tuned: edit/penalty configs are read from the dev-search log.
"""
import json
import sys

import torch

import config as C
from arms import load_split, run_split, summarise
from common import ROOT, SV_SYSTEM, load_model, write_jsonl
from dev_search import Multi
from edits import (ColumnProjectionEdit, ResidualHook, VBiasEdit, language_token_sets, random_direction,
                   residual_means, unembedding_direction)
from common import europarl_sentences
from nll import heldout_sentences, mean_nll

OUT = ROOT / "results" / "test"
DEV = ROOT / "results" / "dev"


def parse_edit(name):
    # loopedit_L15+19_s1.0 / loopedit_L18_s3.0
    _, L, s = name.split("_")
    return [int(x) for x in L[1:].split("+")], float(s[1:])


def edit_ctx(model, layers, s, dirs, frobs=None):
    return Multi(*[ColumnProjectionEdit(model, L, dirs[L], 64, s,
                                        target_frob=None if frobs is None else frobs[L]) for L in layers])


def main(only=None):
    tok, model = load_model()
    log = json.load(open(DEV / "dev_search_log.json"))
    layers, s = parse_edit(log["selected_edit"])
    rp = float(log["selected_reppen"].split("_")[1])
    # conventional activation steering (dev_search2): same selection rule as dev_search.py (see config.py)
    base_en = log["baseline"]["en"]["on_target"]
    ok = [e for e in log.get("configs2", []) if e["sv"]["on_target"] >= 0.9
          and e["en"]["on_target"] >= base_en - C.EN_TOLERANCE]
    steer = min(ok, key=lambda e: (e["sv"]["loop"], e["name"]))["name"] if ok else None
    loop_dirs = {L: torch.load(DEV / f"loop_dir_L{L}.pt") for L in layers}
    loop_frobs = {L: ColumnProjectionEdit(model, L, loop_dirs[L], 64, s).frob for L in layers}
    rand_dirs = {seed: {L: random_direction(model.config.hidden_size, 1000 * seed + L) for L in layers}
                 for seed in C.RANDOM_SEEDS}

    # W0: zero-feedback, weight-only Swedish direction (from lm_head + final norm + token stats)
    sv_ids, en_ids = language_token_sets(tok, C.N_TOKENSET)
    w_dir = unembedding_direction(model, sv_ids, en_ids)
    resid_norm = residual_means(model, tok, europarl_sentences("english", [0])[:50], C.W0_LAYER).norm().item()
    w0 = VBiasEdit(model, C.W0_LAYER, w_dir, C.W0_SCALE * resid_norm)

    arms = {
        "baseline": {},
        "sv_system_prompt": {"system": SV_SYSTEM},
        # T/top-p/rep-penalty as Qwen recommends; NOTE top_k was not applied (Qwen recommends 20)
        "qwen_recommended_sampling": {"do_sample": True, "temperature": 0.7, "top_p": 0.8,
                                      "repetition_penalty": 1.05, "seed": C.SEED},
        **{f"qwen_recommended_sampling_seed{k}": {"do_sample": True, "temperature": 0.7, "top_p": 0.8,
                                                  "repetition_penalty": 1.05, "seed": C.SEED + 1000 * k}
           for k in (1, 2)},
        f"repetition_penalty_{rp}": {"repetition_penalty": rp},
        f"loop_edit[{log['selected_edit']}]": {"ctx": edit_ctx(model, layers, s, loop_dirs)},
        # random controls v1: same weights touched, but much smaller perturbation (review issue)
        **{f"random_edit_seed{k}": {"ctx": edit_ctx(model, layers, s, rand_dirs[k])} for k in C.RANDOM_SEEDS},
        # random controls v2: same columns-count AND same Frobenius norm of the update
        **{f"random_edit_normmatched_seed{k}": {"ctx": edit_ctx(model, layers, s, rand_dirs[k], loop_frobs)}
           for k in C.RANDOM_SEEDS},
        # article-faithful strength (s=1 = removal; selected s=3 is an overshoot); post hoc
        f"loop_edit_s1[loopedit_L{'+'.join(map(str, layers))}_s1.0]": {"ctx": edit_ctx(model, layers, 1.0, loop_dirs)},
        "weight_only_sv_direction_W0": {"ctx": w0},
        **({f"steering[{steer}]": {"ctx": ResidualHook(
            model, int(steer.split("_")[1][1:]),
            -float(steer.split("_")[2][1:]) * torch.load(DEV / f"loop_resdiff_L{steer.split('_')[1][1:]}.pt"))}}
           if steer else {}),
    }
    meta = {"selected_edit": log["selected_edit"], "edited_weights": 896 * 64 * len(layers),
            "selected_reppen": rp, "selected_steering": steer,
            "loop_edit_frob": loop_frobs,
            "random_frob_unmatched": {k: [ColumnProjectionEdit(model, L, rand_dirs[k][L], 64, s).frob for L in layers]
                                      for k in C.RANDOM_SEEDS}, "W0": {"layer": C.W0_LAYER, "scale": C.W0_SCALE,
                                          "resid_norm": resid_norm, "reach_cos": w0.reach,
                                          "n_params": w0.n_params}}
    summ_path = OUT / "summary.json"
    summary = json.load(open(summ_path)) if summ_path.exists() else {}
    summary["meta"] = meta
    sv_test, en_test = load_split("sv_test"), load_split("en_test")
    held = heldout_sentences(C.N_NLL_SENT)
    for name, kw in arms.items():
        if only and name not in only:
            continue
        rs, dt1 = run_split(tok, model, sv_test, **kw)
        re_, dt2 = run_split(tok, model, en_test, **kw)
        write_jsonl(OUT / f"{name}.jsonl", rs + re_)
        entry = {"sv": summarise(rs), "en": summarise(re_), "seconds": round(dt1 + dt2, 1)}
        if "ctx" in kw or name == "baseline":
            with kw.get("ctx") or Multi():
                entry["nll"] = {lang: mean_nll(model, tok, sents) for lang, sents in held.items()}
        summary[name] = entry
        json.dump(summary, open(summ_path, "w"), indent=1, ensure_ascii=False)
        print(name, json.dumps({k: (v if k == "nll" else {m: v[m] for m in ("on_target", "loop", "rep4")})
                                for k, v in entry.items() if k != "seconds"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
