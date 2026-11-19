"""Step 5: tables, paired bootstrap CIs and figures from results/test and results/judging."""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, read_jsonl

T = ROOT / "results" / "test"
J = ROOT / "results" / "judging"
B = 4000


def boot_diff(a, b, seed=0):
    """Paired bootstrap 95% CI of mean(a) - mean(b) (same items, same order)."""
    rng = random.Random(seed)
    n = len(a)
    ds = sorted(sum(a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(B))
    return sum(x - y for x, y in zip(a, b)) / n, ds[int(0.025 * B)], ds[int(0.975 * B)]


def boot_mean(a, seed=0):
    rng = random.Random(seed)
    n = len(a)
    ms = sorted(sum(a[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(a) / n, ms[int(0.025 * B)], ms[int(0.975 * B)]


def check_integrity(grades, key):
    """Fail loudly if a graded sheet item no longer matches the generation it is keyed to
    (e.g. after re-running generation): grades must never attach to different text."""
    sheets = {}
    for p in J.glob("sheet_*.json"):
        for it in json.load(open(p)):
            sheets[it["item_id"]] = it["response"]
    arms_all = json.load(open(J / "KEY_do_not_show_judges.json"))
    amap = {**arms_all.get("arms", {}), **arms_all.get("arms_A2", {})}
    gen = {}
    for f in T.glob("*.jsonl"):
        for r in read_jsonl(f):
            gen[(f.stem, r["id"])] = r["response"]
    bad = 0
    for iid in grades:
        k = key[iid]
        assert iid in sheets, f"graded item {iid} missing from sheets"
        field = "arm2" if "arm2" in k else "arm" if "arm" in k else None
        if field:
            if sheets[iid] != gen[(amap[k[field]], k["prompt_id"])]:
                bad += 1
    assert bad == 0, f"{bad} graded items no longer match their generations - regenerate sheets and re-grade"


def fmt(d, lo, hi):
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main():
    summary = json.load(open(T / "summary.json"))
    meta = summary["meta"]
    arms = [k for k in summary if k != "meta"]
    data = {a: read_jsonl(T / f"{a}.jsonl") for a in arms}
    sv = {a: [r for r in data[a] if r["lang"] == "sv"] for a in arms}
    en = {a: [r for r in data[a] if r["lang"] == "en"] for a in arms}
    for a in arms:
        assert [r["id"] for r in sv[a]] == [r["id"] for r in sv["baseline"]]
    loop = {a: [float(r["loop"]) for r in sv[a]] for a in arms}
    base_nll = summary["baseline"]["nll"]
    L = ["# Held-out test results\n",
         f"Loop edit selected on dev: `{meta['selected_edit']}`, {meta['edited_weights']:,} weights; "
         f"update Frobenius norm {list(meta.get('loop_edit_frob', {'?': float('nan')}).values())[0]:.2f}. "
         f"Unmatched random updates had norm "
         + ", ".join(f"{v[0]:.2f}" for v in meta.get('random_frob_unmatched', {}).values()) + ".\n",
         "## 1. Automatic metrics (80 Swedish, 35 English test prompts; greedy unless stated)\n",
         "| arm | sv: in Swedish | sv: Danish/Norwegian | sv: loop rate | Δ loop vs baseline [95% CI] | identical to baseline | en: in English | en: loop | ΔNLL sv / en / da |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s, e = sv[a], en[a]
        d = boot_diff(loop[a], loop["baseline"])
        same = sum(x["response"] == y["response"] for x, y in zip(data[a], data["baseline"]))
        nll = summary[a].get("nll")
        dn = "/".join(f"{nll[k] - base_nll[k]:+.3f}" for k in ("swedish", "english", "danish")) if nll and a != "baseline" else ("0 (4.240 / 3.751 / 4.476)" if a == "baseline" else "–")
        L.append(f"| {a} | {sum(r['on_target'] for r in s)/len(s):.2f} | {sum(r['label'] in ('da','no') for r in s)/len(s):.2f} | "
                 f"{sum(loop[a])/len(s):.2f} | {fmt(*d) if a != 'baseline' else '–'} | {same}/{len(data[a])} | "
                 f"{sum(r['on_target'] for r in e)/len(e):.2f} | {sum(r['loop'] for r in e)/len(e):.2f} | {dn} |")

    # targeted edit vs random edits
    le = next(a for a in arms if a.startswith("loop_edit["))
    L += ["\n## 2. Targeted edit vs random edits (paired, 80 Swedish prompts; Δ loop rate)\n"]
    for a in arms:
        if a.startswith("random_edit"):
            L.append(f"- {le} − {a}: {fmt(*boot_diff(loop[le], loop[a]))}")
    nm = [a for a in arms if a.startswith("random_edit_normmatched")]
    if nm:
        pooled = [sum(loop[a][i] for a in nm) / len(nm) for i in range(len(loop[le]))]
        L.append(f"- {le} − mean of norm-matched random edits: {fmt(*boot_diff(loop[le], pooled))}")
        L.append(f"- mean of norm-matched random edits − baseline: {fmt(*boot_diff(pooled, loop['baseline']))}")
    s1 = [a for a in arms if a.startswith("loop_edit_s1")]
    if s1 and nm:
        L.append(f"- {s1[0]} (removal, s=1) − baseline: {fmt(*boot_diff(loop[s1[0]], loop['baseline']))}")

    ex = ROOT / "results" / "extra" / "random_normmatched_summary.json"
    if ex.exists() and nm:
        rates = {int(a.rsplit("seed", 1)[1]): sum(loop[a]) / len(loop[a]) for a in nm}
        rates.update({int(k): v["loop"] for k, v in json.load(open(ex)).items()})
        e = sum(loop[le]) / len(loop[le])
        vals = sorted(rates.values())
        p = (sum(v <= e for v in vals) + 1) / (len(vals) + 1)
        L.append(f"\nNull distribution: {len(vals)} random directions, each rescaled to the loop edit's update norm. "
                 f"Swedish loop rates: min {vals[0]:.3f}, median {vals[len(vals)//2]:.3f}, max {vals[-1]:.3f} "
                 f"(baseline {sum(loop['baseline'])/80:.3f}). Loop edit {e:.3f}: {sum(v <= e for v in vals)} of "
                 f"{len(vals)} random edits did as well or better; empirical one-sided p = {p:.3f} = (k+1)/(N+1).")

    samp = [a for a in arms if a.startswith("qwen_recommended_sampling")]
    if len(samp) > 1:
        rates = [sum(loop[a]) / len(loop[a]) for a in samp]
        L.append(f"\nSampling arm over {len(samp)} seeds: Swedish loop rate {', '.join(f'{x:.3f}' for x in rates)} "
                 f"(mean {sum(rates)/len(rates):.3f}); in-Swedish rate "
                 + ", ".join(f"{sum(r['on_target'] for r in sv[a])/80:.2f}" for a in samp) + ".")

    # ---- judging
    grades = {}
    for p in sorted(J.glob("grades_*.json")):
        for g in json.load(open(p)):
            grades[g["item_id"]] = g
    key = json.load(open(J / "KEY_do_not_show_judges.json"))["key"] if grades else {}
    check_integrity(grades, key)
    judged = {}
    en_ref = None

    def table(field, title, note):
        by = {}
        for iid, g in grades.items():
            k = key.get(iid, {})
            if field in k:
                by.setdefault(k[field], {})[k["prompt_id"]] = g
        if not by:
            return by
        L.extend([f"\n## {title}\n", note + "\n",
                  "| arm | fluency [95% CI] | adequacy [95% CI] | Δ fluency vs baseline (paired) | Δ adequacy vs baseline (paired) | usable answers (adequacy 2) |",
                  "|---|---|---|---|---|---|"])
        order = ["baseline"] + [a for a in by if a != "baseline"]
        for a in order:
            gs = by[a]
            ps = sorted(gs)
            f = [gs[p]["fluency"] for p in ps]
            ad = [gs[p]["adequacy"] for p in ps]
            if a == "baseline" or "baseline" not in by:
                df = da = "–"
            else:
                c = sorted(set(gs) & set(by["baseline"]))
                df = fmt(*boot_diff([gs[p]["fluency"] for p in c], [by["baseline"][p]["fluency"] for p in c]))
                da = fmt(*boot_diff([gs[p]["adequacy"] for p in c], [by["baseline"][p]["adequacy"] for p in c]))
            fm, fl, fh = boot_mean(f)
            am, al, ah = boot_mean(ad)
            L.append(f"| {a} | {fm:.2f} [{fl:.2f}, {fh:.2f}] | {am:.2f} [{al:.2f}, {ah:.2f}] | {df} | {da} | {sum(x == 2 for x in ad)}/{len(ad)} |")
        return by

    byA2 = table("arm2", "3. Blind quality grading, one judge for all arms (Sheet A2: 40 Swedish test prompts per arm)",
                 "Scores 0–2. Fluency: grammatical, natural Swedish. Adequacy: correct, relevant, helpful. One fresh LLM judge (judge 4) graded every item, shuffled, with no arm names.")
    byA = table("arm", "3b. First grading round (Sheet A: 6 arms, judge 1; Sheet C: sampling arm, judge 3)",
                "Kept for transparency. Different judges graded different arms here, which Sheet A2 fixes.")
    if byA2:
        for name, by in (("A2", byA2),):
            order = ["baseline"] + sorted(a for a in by if a != "baseline")
            judged = {a: sum(g["adequacy"] for g in by[a].values()) / len(by[a]) for a in order}
        # agreement between rounds on identical (arm, prompt) items
        pairs = []
        for a2, a1 in (("baseline", "baseline"), ("loop_edit", "loop_edit"), ("sv_system_prompt", "sv_system_prompt"),
                       ("reppen", "reppen"), ("steering", "steering"), ("qwen_sampling", "qwen_sampling")):
            if a2 in byA2 and a1 in byA:
                for p in byA2[a2]:
                    if p in byA[a1]:
                        pairs.append((byA2[a2][p], byA[a1][p]))
        if pairs:
            L.append(f"\nAgreement between the A2 judge and the first-round judges on {len(pairs)} identical items: "
                     f"fluency {sum(x['fluency'] == y['fluency'] for x, y in pairs)/len(pairs):.2f}, "
                     f"adequacy {sum(x['adequacy'] == y['adequacy'] for x, y in pairs)/len(pairs):.2f} (exact match).")
        reps = [(grades[i], grades[k["repeat_of"]]) for i, k in key.items() if "repeat_of" in k and i in grades]
        if reps:
            L.append(f"Agreement of judge 3 re-grading 40 Sheet-A items: fluency "
                     f"{sum(a['fluency'] == b['fluency'] for a, b in reps)/len(reps):.2f}, adequacy "
                     f"{sum(a['adequacy'] == b['adequacy'] for a, b in reps)/len(reps):.2f}. "
                     "Exact agreement is inflated by the floor effect: most items score fluency 1 and adequacy 0.")
        # does looping explain low adequacy? baseline, A2
        bl = byA2.get("baseline", {})
        lp = {r["id"]: r["loop"] for r in sv["baseline"]}
        lo = [g["adequacy"] for p, g in bl.items() if lp[p]]
        nl = [g["adequacy"] for p, g in bl.items() if not lp[p]]
        if lo and nl:
            L.append(f"\nBaseline answers that loop: mean adequacy {sum(lo)/len(lo):.2f} (n={len(lo)}); answers that don't loop: "
                     f"{sum(nl)/len(nl):.2f} (n={len(nl)}).")
        amap2 = json.load(open(J / "KEY_do_not_show_judges.json"))["arms_A2"]
        lp_all, nl_all = [], []
        for a, gs in byA2.items():
            lpa = {r["id"]: r["loop"] for r in read_jsonl(T / f"{amap2[a]}.jsonl")}
            for p, g in gs.items():
                (lp_all if lpa[p] else nl_all).append(g["adequacy"])
        L.append(f"Pooled over all A2 arms: answers that loop {sum(lp_all)/len(lp_all):.2f} (n={len(lp_all)}), "
                 f"answers that don't {sum(nl_all)/len(nl_all):.2f} (n={len(nl_all)}), "
                 f"{sum(x == 2 for x in nl_all)} of the {len(nl_all)} loop-free answers fully adequate.")

    # cross-lingual
    def gap(field, title, extra=""):
        pairs = {}
        for iid, g in grades.items():
            k = key.get(iid, {})
            if field in k:
                pairs.setdefault(k["pair"], {})[k[field]] = g
        full = {p: v for p, v in pairs.items() if "sv" in v and "en" in v}
        if not full:
            return None
        sv_a = [v["sv"]["adequacy"] for v in full.values()]
        en_a = [v["en"]["adequacy"] for v in full.values()]
        known = [v for v in full.values() if v["en"]["adequacy"] == 2]
        L.extend([f"\n## {title}\n", extra,
                  f"- Mean adequacy: English {sum(en_a)/len(en_a):.2f}, Swedish {sum(sv_a)/len(sv_a):.2f}; "
                  f"difference {fmt(*boot_diff(en_a, sv_a))} (n={len(full)} pairs)",
                  f"- Mean fluency: English {sum(v['en']['fluency'] for v in full.values())/len(full):.2f}, "
                  f"Swedish {sum(v['sv']['fluency'] for v in full.values())/len(full):.2f}",
                  f"- {sum(v['sv']['adequacy'] == 2 for v in known)} of the {len(known)} questions answered fully adequately in English were also adequate in Swedish"])
        return sum(en_a) / len(en_a)

    cl = ROOT / "results" / "crosslingual" / "pairs_256tok.jsonl"
    extra = ""
    if cl.exists():
        rows = read_jsonl(cl)
        w = {l: [r["n_words"] for r in rows if r["lang"] == l] for l in ("sv", "en")}
        extra = (f"Answers at 256 new tokens: mean {sum(w['sv'])/50:.0f} words in Swedish and {sum(w['en'])/50:.0f} in English. "
                 f"Loop rate: Swedish {sum(r['loop'] for r in rows if r['lang']=='sv')/50:.2f}, English {sum(r['loop'] for r in rows if r['lang']=='en')/50:.2f}.\n")
    en_ref = gap("lang2", "4. Cross-lingual gap: the same 50 questions in English and Swedish, baseline model, 256-token budget (Sheet B2, judge 5)", extra)
    cl3 = ROOT / "results" / "crosslingual" / "pairs_256tok_sampling.jsonl"
    extra3 = ""
    if cl3.exists():
        rows3 = read_jsonl(cl3)
        extra3 = (f"Sampling (T=0.7, top-p 0.8, repetition penalty 1.05, top-k off), 256 tokens. Loop rate: Swedish "
                  f"{sum(r['loop'] for r in rows3 if r['lang']=='sv')/50:.2f}, English {sum(r['loop'] for r in rows3 if r['lang']=='en')/50:.2f}; "
                  f"mean words {sum(r['n_words'] for r in rows3 if r['lang']=='sv')/50:.0f} / {sum(r['n_words'] for r in rows3 if r['lang']=='en')/50:.0f}.\n")
    gap("lang3", "4a. Cross-lingual gap with sampling, loop rates equal across languages (Sheet B3, judge 6)", extra3)
    r1 = gap("lang", "4b. First version (Sheet B, judge 2, 96-token budget; Swedish answers about a third shorter)")
    en_ref = en_ref if en_ref is not None else r1

    (ROOT / "results" / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # ---- figure
    show = [a for a in arms if not a.startswith("random_edit_seed") and not a.startswith("qwen_recommended_sampling_seed")]
    fig, axes = plt.subplots(1, 2 if judged else 1, figsize=(13.5 if judged else 7, 4.8))
    axes = axes if judged else [axes]
    names = [a.split("[")[0] for a in show]
    axes[0].barh(names, [sum(loop[a]) / len(loop[a]) for a in show], color="#4a6fa5")
    axes[0].set_xlabel("Share of Swedish test answers that loop (80 prompts)")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    if judged:
        ks = list(judged)
        axes[1].barh(ks, [judged[k] for k in ks], color="#c0504d")
        axes[1].set_xlabel("Blind-judged adequacy (0–2), 40 Swedish prompts")
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 2)
        if en_ref is not None:
            axes[1].axvline(en_ref, color="k", ls="--", lw=1)
            axes[1].text(en_ref + 0.03, len(ks) - 0.6, f"baseline, same 50 questions\nasked in English\n(256-token answers): {en_ref:.2f}",
                         fontsize=8, va="bottom")
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "fig_main.png", dpi=150)


if __name__ == "__main__":
    main()
