"""Convert the Ollama checkpoint used in this study to a Transformers folder.

The study's model is Ollama `qwen2.5:0.5b-instruct-fp16` (an fp16 GGUF export of
Qwen/Qwen2.5-0.5B-Instruct), GGUF blob sha256
6f96e01a3f550ca08aea1e5725bb8d5a7eccc6f281c30417e9d380b8c46467bd.
Get it with `ollama pull qwen2.5:0.5b-instruct-fp16`; the blob is at
~/.ollama/models/blobs/sha256-6f96e01a... . fp16 -> fp32 is lossless, so this should
match the Hugging Face weights up to Ollama's bf16->fp16 export (which can round
values; we could not verify this against the HF release). Ollama manifest: library/qwen2.5,
tag 0.5b-instruct-fp16, config digest sha256:ff54cb6e5adea7664c2b555d4bbaf5ee5acbb325104fd8db41282cfb866d0c2e.
If you load Qwen/Qwen2.5-0.5B-Instruct from HF
directly instead, results may differ slightly (fp16 rounding, chat template).

usage: python scripts/convert_gguf.py /path/to/blob_or.gguf model/Qwen2.5-0.5B-Instruct
"""
import hashlib
import os
import shutil
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
h = hashlib.sha256(open(src, "rb").read()).hexdigest()
print("sha256", h)
tmp = tempfile.mkdtemp()
shutil.copy(src, os.path.join(tmp, "model.gguf"))
tok = AutoTokenizer.from_pretrained(tmp, gguf_file="model.gguf")
m = AutoModelForCausalLM.from_pretrained(tmp, gguf_file="model.gguf", dtype=torch.float32)
cfg = m.config
if hasattr(cfg, "quantization_config"):
    delattr(cfg, "quantization_config")
m2 = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
res = m2.load_state_dict(m.state_dict(), strict=False)
# lm_head is tied to embed_tokens, so it may be reported missing; nothing else may be.
assert not res.unexpected_keys and all(k == "lm_head.weight" for k in res.missing_keys), res
m2.tie_weights()
m2.save_pretrained(out)
tok.save_pretrained(out)
print("saved", out)
