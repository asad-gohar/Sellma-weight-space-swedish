"""Cross-lingual gap with a longer budget (review fix).

The first gap analysis (Sheet B) used 96 new tokens. Swedish needs more tokens per
word, so Swedish answers were shorter (~51 vs ~77 words). Here both languages get
256 new tokens, and the answer length in words is recorded. Baseline model, greedy.
"""
import re

from build_pairs import PAIRS
from common import ROOT, generate, load_model, score_response, write_jsonl

tok, model = load_model()
sv = [p[0] for p in PAIRS]
en = [p[1] for p in PAIRS]
rows = []
for lang, prompts in (("sv", sv), ("en", en)):
    outs = generate(tok, model, prompts, max_new_tokens=256, batch_size=10)
    for k, (p, o) in enumerate(zip(prompts, outs)):
        rows.append({"pair": k, "lang": lang, "prompt": p, "response": o,
                     "n_words": len(re.findall(r"\w+", o)), **score_response(o, lang)})
write_jsonl(ROOT / "results" / "crosslingual" / "pairs_256tok.jsonl", rows)
for lang in ("sv", "en"):
    r = [x for x in rows if x["lang"] == lang]
    print(lang, "mean words %.1f" % (sum(x["n_words"] for x in r) / len(r)),
          "loop %.2f" % (sum(x["loop"] for x in r) / len(r)))
