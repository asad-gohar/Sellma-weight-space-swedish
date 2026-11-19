"""Assemble report_src/*.md + results into report.pdf (pandoc + headless Chrome)."""
import json
import re
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[1]
S = R / "report_src"


def devtable(path, title, steer=None):
    L = json.load(open(path))
    rows = L["configs"] + L.get("configs2", [])
    out = [f"**{title}** (dev: 30 Swedish / 15 English prompts; baseline Swedish loop {L['baseline']['sv']['loop']:.2f}, "
           f"Swedish-answer rate {L['baseline']['sv']['on_target']:.2f})\n",
           "| config | sv loop | sv answered in Swedish | en loop | en answered in English |", "|---|---|---|---|---|"]
    for e in rows:
        out.append(f"| {e['name']} | {e['sv']['loop']:.2f} | {e['sv']['on_target']:.2f} | {e['en']['loop']:.2f} | {e['en']['on_target']:.2f} |")
    if "selected_edit" in L:
        out.append(f"\nSelected by the rule in `src/config.py`: `{L['selected_edit']}`, `{L['selected_reppen']}`"
                   + (f", steering `{steer}`." if steer else "."))
    return "\n".join(out)


def examples():
    arms = ["baseline", "qwen_recommended_sampling", "repetition_penalty_1.2", "loop_edit[loopedit_L15_s3.0]",
            "random_edit_normmatched_seed2", "steering[steer_L18_b1.0]", "weight_only_sv_direction_W0"]
    D = {a: {r["id"]: r for r in map(json.loads, open(R / "results" / "test" / f"{a}.jsonl"))} for a in arms}
    ids = [i for i in D["baseline"] if i.startswith("sv")]
    ex = ["Held-out test; the first ~200 characters of each answer, unedited.\n"]
    for i in [ids[8], ids[16], ids[39]]:
        ex.append(f"*Prompt:* „{D['baseline'][i]['prompt']}“\n")
        for a in arms:
            t = D[a][i]["response"].replace("\n", " ").replace("|", "/")[:200]
            ex.append(f"- **{a.split('[')[0]}** (loop={D[a][i]['loop']}): {t}…")
        ex.append("")
    return "\n".join(ex)


def main():
    meta = json.load(open(R / "results" / "test" / "summary.json"))["meta"]
    body = (S / "body.md").read_text()
    summ, rest = body.split("## 2. Hypothesis", 1)
    rest = "## 2. Hypothesis" + rest
    tables = (R / "results" / "tables.md").read_text().split("\n")
    tables = "\n".join(l for l in tables if not l.startswith("# Held-out") and not l.startswith("Selected configs"))
    tables = re.sub(r"^## (\d+)([ab]?)\. ", lambda m: f"### 4.{m.group(1)}{m.group(2)} ", tables, flags=re.M)
    cfg = (f"Configurations selected on dev: loop edit `{meta['selected_edit']}` ({meta['edited_weights']:,} weights), "
           f"repetition penalty {meta['selected_reppen']}, steering `{meta['selected_steering']}`; logit-lens edit at layer "
           f"{meta['W0']['layer']} ({meta['W0']['n_params']} bias values, reach cosine {meta['W0']['reach_cos']:.2f}). "
           "In the tables, `weight_only_sv_direction_W0` is the logit-lens heuristic edit, `random_edit_seed*` are the "
           "first random controls (matched in weights touched only) and `random_edit_normmatched_seed*` the final ones "
           "(also matched in update norm).\n")
    doc = f"""---
title: "Is Swedish degeneration in a small open model a removable obstacle?"
subtitle: "Testing the weight-space-programming claim for SE.LLMA, candidate assignment report"
author: "Asad Gohar"
date: "November 2026"
---

{summ}
{(S / 'critique.md').read_text()}

{rest}
{cfg}
{tables}

![Left: share of the 80 Swedish test answers that loop, per arm. Right: blind-judged adequacy (0–2) on 40 Swedish test prompts, one judge for all arms. The dashed line is the baseline model on the 50 paired questions asked in English (256-token answers), so it covers a different, overlapping prompt set.](../results/fig_main.png){{width=100%}}

### 4.5 Examples

{examples()}

{(S / 'body2.md').read_text()}

{devtable(R / 'results/dev/dev_search_log.json', 'Table D1. Dev search (final, Swedish-only directions)', meta.get('selected_steering'))}

{devtable(R / 'results/dev_v1_confounded/dev_search_log.json', 'Table D2. First dev search (confounded: directions built from Swedish + English answers)')}

### AI assistance
I carried out this work with substantial help from an AI assistant (Claude), which wrote most of the code and drafted text under my direction. Answer quality was graded blind by LLM judges. Details are in `AI_USAGE.md` in the repository.
"""
    (S / "report.md").write_text(doc)
    subprocess.run(["pandoc", "report.md", "-s", "-c", "style.css", "--metadata", "pagetitle=SE.LLMA report",
                    "-o", "report.html"], cwd=S, check=True)
    subprocess.run(["google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={R / 'report.pdf'}", "report.html"], cwd=S, check=True,
                   stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
