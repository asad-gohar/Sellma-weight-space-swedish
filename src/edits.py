"""Edit directions and the exact v_proj-bias weight edit.

Why v_proj bias: in Qwen2 the attention value projection has a bias b_v. Because
attention weights sum to 1 over positions, adding delta_b to b_v adds the constant
vector  W_o @ expand(delta_b)  to the residual stream at every position of that
layer. So a steering vector can be written into the checkpoint as a genuine,
inspectable weight edit (128 numbers per layer for Qwen2.5-0.5B), with no hooks.
The catch: expand() repeats each of the 2 KV heads' 64-dim bias over its 7 query
heads, so only a 128-dim subspace of the 896-dim residual is reachable per layer.
We report how much of the requested vector is reachable ("reach" = cosine).
"""
from __future__ import annotations

import collections
import math

import torch

from common import europarl_sentences

# Europarl files 0-5 are used to build directions / token sets ("dev text");
# files 7-9 are held out for the NLL regression check. File 6 is unused.
DIRECTION_FILES = list(range(0, 6))
HELDOUT_FILES = [7, 8, 9]


# ------------------------------------------------------------ token statistics
def language_token_sets(tok, n_top=300, min_count=20):
    """Tokens characteristic of Swedish vs English, by corpus log-ratio.

    Uses tokenizer + Europarl text only: no forward pass of the model.
    """
    counts = {}
    for lang in ("swedish", "english"):
        c = collections.Counter()
        for s in europarl_sentences(lang, DIRECTION_FILES):
            c.update(tok(" " + s, add_special_tokens=False)["input_ids"])
        counts[lang] = c
    tot = {k: sum(v.values()) for k, v in counts.items()}

    def lr(t, a, b):
        return math.log((counts[a][t] + 1) / tot[a]) - math.log((counts[b][t] + 1) / tot[b])

    vocab = set(counts["swedish"]) | set(counts["english"])
    sv = [t for t in vocab if counts["swedish"][t] >= min_count]
    en = [t for t in vocab if counts["english"][t] >= min_count]
    sv = sorted(sv, key=lambda t: -lr(t, "swedish", "english"))[:n_top]
    en = sorted(en, key=lambda t: -lr(t, "english", "swedish"))[:n_top]
    return sv, en


# ------------------------------------------------------------ directions
def unembedding_direction(model, sv_ids, en_ids):
    """Weight-only direction: what residual change most raises Swedish-token logits
    relative to English-token logits, under the logit-lens assumption.

    logits = W_U (g * h / rms(h))  =>  d logit-gap / d h  ∝  g * (mean W_U[sv] - mean W_U[en]).
    """
    W_U = model.lm_head.weight.detach().float()
    g = model.model.norm.weight.detach().float()
    d = W_U[sv_ids].mean(0) - W_U[en_ids].mean(0)
    v = g * d
    return v / v.norm()


@torch.no_grad()
def residual_means(model, tok, sentences, layer, batch=16, max_len=64):
    """Mean residual-stream state after block `layer`, over all tokens."""
    acc, n = None, 0
    tok.padding_side = "right"
    for i in range(0, len(sentences), batch):
        enc = tok(sentences[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer + 1].float()  # hidden_states[0] = embeddings
        m = enc["attention_mask"].unsqueeze(-1).float()
        s = (h * m).sum((0, 1))
        acc = s if acc is None else acc + s
        n += m.sum().item()
    tok.padding_side = "left"
    return acc / n


def random_direction(dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


# ------------------------------------------------------------ the weight edit
def _expand_matrix(model):
    cfg = model.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_h
    rep = n_h // n_kv
    E = torch.zeros(n_h * hd, n_kv * hd)
    for h in range(n_h):
        kv = h // rep
        E[h * hd:(h + 1) * hd, kv * hd:(kv + 1) * hd] = torch.eye(hd)
    return E


def vbias_delta_for(model, layer, target):
    """Least-squares delta_b so that W_o @ expand(delta_b) ≈ target. Returns
    (delta_b, achieved residual vector, reach cosine)."""
    attn = model.model.layers[layer].self_attn
    M = attn.o_proj.weight.detach().float() @ _expand_matrix(model)  # [d_model, n_kv*hd]
    sol = torch.linalg.lstsq(M, target.float().unsqueeze(1)).solution.squeeze(1)
    achieved = M @ sol
    reach = torch.nn.functional.cosine_similarity(achieved, target.float(), dim=0).item()
    return sol, achieved, reach


class VBiasEdit:
    """Context manager that applies, then exactly reverts, a v_proj-bias edit."""

    def __init__(self, model, layer, direction, scale):
        self.model, self.layer = model, layer
        self.delta, self.achieved, self.reach = vbias_delta_for(model, layer, direction * scale)
        self.n_params = self.delta.numel()

    def __enter__(self):
        b = self.model.model.layers[self.layer].self_attn.v_proj.bias
        self._orig = b.detach().clone()
        with torch.no_grad():
            b.add_(self.delta.to(b.dtype))
        return self

    def __exit__(self, *a):
        with torch.no_grad():
            self.model.model.layers[self.layer].self_attn.v_proj.bias.copy_(self._orig)


class ResidualHook:
    """Unconstrained activation steering (adds vector after a block). Not a weight
    edit; included as the conventional steering comparison."""

    def __init__(self, model, layer, vec):
        self.model, self.layer, self.vec = model, layer, vec

    def __enter__(self):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + self.vec.to(out[0].dtype),) + tuple(out[1:])
            return out + self.vec.to(out.dtype)
        self.h = self.model.model.layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


# ------------------------------------------------------------ article-style loop edit
def token_loop_labels(ids, n=8):
    """1 if the token n-gram ending at position t already occurred earlier."""
    seen, lab = set(), []
    for t in range(len(ids)):
        if t + 1 < n:
            lab.append(0)
            continue
        g = tuple(ids[t + 1 - n:t + 1])
        lab.append(int(g in seen))
        seen.add(g)
    return lab


@torch.no_grad()
def loop_direction(model, tok, chats, responses, layer):
    """Mean MLP output at `layer` on looping tokens minus non-looping response tokens.

    Needs forward passes over the model's own (dev/train) generations: this is the
    'feedback' regime, like the article's Qwen experiment, not weight-only reading.
    """
    store = {}
    h = model.model.layers[layer].mlp.register_forward_hook(
        lambda m, i, o: store.__setitem__("o", o.detach()))
    sums = {0: torch.zeros(model.config.hidden_size), 1: torch.zeros(model.config.hidden_size)}
    counts = {0: 0, 1: 0}
    try:
        for chat, resp in zip(chats, responses):
            p_ids = tok(chat, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp, add_special_tokens=False)["input_ids"]
            if len(r_ids) < 10:
                continue
            ids = torch.tensor([p_ids + r_ids])
            model(input_ids=ids)
            o = store["o"][0, len(p_ids):].float()
            for t, l in enumerate(token_loop_labels(r_ids)):
                sums[l] += o[t]
                counts[l] += 1
    finally:
        h.remove()
    if counts[1] == 0:
        raise RuntimeError("no looping tokens found")
    d = sums[1] / counts[1] - sums[0] / counts[0]
    return d / d.norm(), counts


class ColumnProjectionEdit:
    """Remove direction d from the k down_proj output columns that write most along d
    (W[:, j] <- W[:, j] - s * d d^T W[:, j]). Mirrors the article's Qwen edit
    ('subtracted that direction from 64 output columns'). Reverts exactly on exit."""

    def __init__(self, model, layer, direction, k=64, strength=1.0, target_frob=None):
        """strength=1 removes the d-component of the chosen columns; strength=3 flips it
        to -2x (an overshoot). target_frob: rescale the update to this Frobenius norm
        (used to give random-direction controls the same perturbation size)."""
        self.model, self.layer, self.k, self.s = model, layer, k, strength
        W = model.model.layers[layer].mlp.down_proj.weight.detach().float()  # [d_model, d_ff]
        d = direction.float() / direction.norm()
        coef = d @ W                                   # [d_ff]
        self.cols = torch.topk(coef.abs(), k).indices
        self.update = -strength * torch.outer(d, coef[self.cols])  # [d_model, k]
        if target_frob is not None:
            self.update = self.update * (target_frob / self.update.norm())
        self.frob = self.update.norm().item()
        self.n_params = W.shape[0] * k

    def __enter__(self):
        W = self.model.model.layers[self.layer].mlp.down_proj.weight
        self._orig = W[:, self.cols].detach().clone()
        with torch.no_grad():
            W[:, self.cols] += self.update.to(W.dtype)
        return self

    def __exit__(self, *a):
        with torch.no_grad():
            self.model.model.layers[self.layer].mlp.down_proj.weight[:, self.cols] = self._orig


@torch.no_grad()
def loop_residual_difference(model, tok, chats, responses, layers):
    """Raw (unnormalised) mean residual after each block in `layers` on looping
    tokens minus non-looping response tokens. Subtracting it (beta=1) moves the
    mean looping state onto the mean non-looping state: a principled scale."""
    H = model.config.hidden_size
    sums = {L: {0: torch.zeros(H), 1: torch.zeros(H)} for L in layers}
    counts = {0: 0, 1: 0}
    for chat, resp in zip(chats, responses):
        p_ids = tok(chat, add_special_tokens=False)["input_ids"]
        r_ids = tok(resp, add_special_tokens=False)["input_ids"]
        if len(r_ids) < 10:
            continue
        hs = model(input_ids=torch.tensor([p_ids + r_ids]), output_hidden_states=True).hidden_states
        labs = token_loop_labels(r_ids)
        lab_t = torch.tensor(labs)
        for L in layers:
            h = hs[L + 1][0, len(p_ids):].float()
            for l in (0, 1):
                sums[L][l] += h[lab_t == l].sum(0)
        counts[0] += int((lab_t == 0).sum()); counts[1] += int((lab_t == 1).sum())
    return {L: sums[L][1] / counts[1] - sums[L][0] / counts[0] for L in layers}, counts
