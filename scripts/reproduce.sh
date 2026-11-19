#!/usr/bin/env bash
# Reproduce the experiment end to end on CPU (~5.5 h on 2 cores).
# Requires the model in model/Qwen2.5-0.5B-Instruct (see README "Model").
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -r requirements.txt
python data/build_prompts.py                 # fixed-seed dev/test split (already committed)
[ -d data/europarl_raw ] || { curl -sSL -o /tmp/ep.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip && unzip -q /tmp/ep.zip -d data/; }
cd src
python probe_baseline.py                     # dev: characterise the failure
python dev_search.py --mixed-lang            # dev: FIRST (confounded) search, kept for transparency
python dev_search2.py --mixed-lang           #      (the original run of this step was stopped after 2 configs)
python dev_search.py                         # dev: article-style column edit + repetition-penalty search
python dev_search2.py                        # dev: residual-steering search
python run_test.py                           # held-out test: all arms + Europarl NLL
python crosslingual.py                       # 50 sv/en pairs at 256 tokens
python extra_controls.py                     # 2nd review: 12 more random edits, sampled pairs, sanity check (~1.5 h)
python prepare_judging.py                    # blind grading sheets A, B, C, A2, B2, B3 + key
# Grading: give each sheet to a fresh LLM or a human grader using results/judging/judge_prompts.md,
# and save results/judging/grades_<sheet>.json as [{"item_id", "fluency", "adequacy"}].
python analyse.py                            # tables, bootstrap CIs, figure
cd .. && python scripts/build_report.py      # report.pdf (needs pandoc + Chrome)
