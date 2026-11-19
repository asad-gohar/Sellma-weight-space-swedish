"""Shared helpers: model loading, batched chat generation, language ID, Europarl text."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "model" / "Qwen2.5-0.5B-Instruct"))
EUROPARL_DIR = Path(os.environ.get("EUROPARL_DIR", ROOT / "data" / "europarl_raw"))
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SV_SYSTEM = "Du är en hjälpsam assistent. Svara alltid på svenska."

torch.set_num_threads(max(1, os.cpu_count() or 1))


def load_model(dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype)
    model.eval()
    return tok, model


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_chat(tok, prompt, system=DEFAULT_SYSTEM):
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts, system=DEFAULT_SYSTEM, max_new_tokens=64, batch_size=10,
             do_sample=False, temperature=1.0, top_p=1.0, seed=0, repetition_penalty=1.0):
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [build_chat(tok, p, system) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True)
        if do_sample:
            torch.manual_seed(seed + i)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None, top_k=None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


# ---------------------------------------------------------------- language ID
_DETECTOR = None
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def detector():
    global _DETECTOR
    if _DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder
        langs = [Language.SWEDISH, Language.ENGLISH, Language.DANISH, Language.BOKMAL,
                 Language.NYNORSK, Language.GERMAN, Language.DUTCH, Language.CHINESE]
        _DETECTOR = LanguageDetectorBuilder.from_languages(*langs).build()
    return _DETECTOR


def lang_of(text: str) -> str:
    """Coarse label: sv / en / da / no / cjk / other / empty."""
    t = text.strip()
    if len(t) < 3:
        return "empty"
    if len(CJK.findall(t)) > 0.2 * len(t):
        return "cjk"
    lang = detector().detect_language_of(t)
    if lang is None:
        return "other"
    name = lang.name
    return {"SWEDISH": "sv", "ENGLISH": "en", "DANISH": "da", "BOKMAL": "no",
            "NYNORSK": "no"}.get(name, "other")


def sentence_labels(text: str):
    sents = [s.strip() for s in re.split(r"(?<=[.!?:\n])\s+", text) if len(s.strip()) >= 15]
    return [lang_of(s) for s in sents]


def repetition_stats(words, n=4, loop_n=5):
    """rep4: share of word 4-grams that already occurred earlier in the response.
    loop: some word 5-gram occurs at least twice (verbatim looping)."""
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen, dup = set(), 0
    for g in grams:
        dup += g in seen
        seen.add(g)
    lg = [tuple(words[i:i + loop_n]) for i in range(len(words) - loop_n + 1)]
    return {"rep4": dup / len(grams) if grams else 0.0, "loop": len(lg) != len(set(lg))}


def score_response(text: str, target: str) -> dict:
    labs = sentence_labels(text)
    whole = lang_of(text)
    frac = (sum(l == target for l in labs) / len(labs)) if labs else float(whole == target)
    words = re.findall(r"\w+", text.lower())
    distinct = len(set(words)) / max(1, len(words))
    rep = repetition_stats(words)
    return {
        **rep,
        "label": whole,
        "on_target": whole == target,
        "pure": whole == target and all(l == target for l in labs),
        "sent_frac": frac,
        "has_cjk": bool(CJK.search(text)),
        "distinct_word_ratio": distinct,
    }


# ---------------------------------------------------------------- Europarl
def europarl_sentences(lang: str, files=None):
    """lang in {swedish, english, danish}. Returns detokenised sentences."""
    d = EUROPARL_DIR / lang
    names = sorted(os.listdir(d))
    if files is not None:
        names = [names[i] for i in files]
    sents = []
    for n in names:
        for line in open(d / n, encoding="utf-8"):
            s = line.strip()
            if len(s) < 40 or len(s) > 300:
                continue
            s = re.sub(r" ([.,;:!?)%])", r"\1", s)
            s = re.sub(r"\( ", "(", s).replace(' " ', ' "')
            sents.append(s)
    return sents
