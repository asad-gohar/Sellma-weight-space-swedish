"""Step 1: characterise the failure on DEV prompts only (baseline + prompting)."""
import json

from arms import load_split, run_split, summarise
from common import load_model, write_jsonl, SV_SYSTEM, ROOT

tok, model = load_model()
out = {}
for split in ("sv_dev", "en_dev"):
    rows = load_split(split)
    for arm, kw in {"baseline": {}, "sv_system": {"system": SV_SYSTEM}}.items():
        if split == "en_dev" and arm == "sv_system":
            continue
        res, dt = run_split(tok, model, rows, **kw)
        write_jsonl(ROOT / "results" / "dev" / f"{split}__{arm}.jsonl", res)
        out[f"{split}/{arm}"] = {**summarise(res), "seconds": round(dt, 1)}
        print(split, arm, json.dumps(out[f"{split}/{arm}"], ensure_ascii=False))
json.dump(out, open(ROOT / "results" / "dev" / "probe_summary.json", "w"), indent=1)
