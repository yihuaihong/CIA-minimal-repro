"""SINGLE SOURCE OF TRUTH for tokenizer loading + chat templating across
training, probe feature extraction, eval and rewards.

Project convention (user decision 2026-08-25): NO thinking mode for Qwen3.
`apply_chat_template(..., enable_thinking=False)` inserts an empty
`<think>\\n\\n</think>\\n\\n` block after the assistant header — that block is
therefore part of EVERY prefix (generation, probe training, probe read-out,
reward). Loading a tokenizer any other way for Qwen3 puts hidden states off the
probe's training distribution (AUDIT_hint_mult_2026-08-25 #6).
"""
from __future__ import annotations
import os
from transformers import AutoTokenizer

QWEN3_THINKING = False   # project-wide; do not override per script


def is_qwen3(name_or_path: str) -> bool:
    return "qwen3" in os.path.basename(str(name_or_path).rstrip("/")).lower()


def patch_tokenizer(tok, name_or_path: str):
    """Idempotent: fixes pad token and forces the Qwen3 thinking convention."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_qwen3(name_or_path) and not getattr(tok, "_cia_patched", False):
        _orig = tok.apply_chat_template

        def _fixed(*a, **k):
            k["enable_thinking"] = QWEN3_THINKING
            return _orig(*a, **k)
        tok.apply_chat_template = _fixed
        tok._cia_patched = True
    return tok


def load_tokenizer(name_or_path: str, **kw):
    """`name_or_path` may be a model name under $SCRATCH/transformers or a path."""
    p = name_or_path if os.path.exists(name_or_path) else f"{os.environ['SCRATCH']}/transformers/{name_or_path}"
    return patch_tokenizer(AutoTokenizer.from_pretrained(p, **kw), name_or_path)


def user_chat(tok, content: str, add_generation_prompt: bool = True) -> str:
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=add_generation_prompt)
