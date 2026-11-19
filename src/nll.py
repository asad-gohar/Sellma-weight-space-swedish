"""Regression check: mean per-token NLL on held-out Europarl text (sv / en / da)."""
import torch

from common import europarl_sentences
from edits import HELDOUT_FILES


def heldout_sentences(n):
    return {lang: europarl_sentences(lang, HELDOUT_FILES)[:n]
            for lang in ("swedish", "english", "danish")}


@torch.no_grad()
def mean_nll(model, tok, sentences, batch=16, max_len=96):
    tok.padding_side = "right"
    tot, n = 0.0, 0
    for i in range(0, len(sentences), batch):
        enc = tok(sentences[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        logits = model(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        m = enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tot += -(lp[m]).sum().item()
        n += m.sum().item()
    tok.padding_side = "left"
    return tot / n
