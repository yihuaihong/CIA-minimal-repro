"""SINGLE SOURCE OF TRUTH for the Hint B_INT probe read-out (C-position).

Definition (S-HINT-PROBE-v2 / S-HINT-EVAL-v2):
  feature  = hidden state at the token holding the answer LETTER after `<mc>`
             in chat(prompt) + generation, at the probe's layer; if no `<mc>`
             letter exists the sample has NO C-position → B_INT is undefined
             (callers decide: eval drops the row, reward treats it as format
             failure). No silent "last token" fallback.
  B_INT    = argmax of a frozen Linear(H,2) on (h - mean) / (std + 1e-8)
  prompt   = chat-templated with CPF_utils.chat (Qwen3: empty think block).
Used by: probe training (scripts/train_hint_cpos_probe_v2.py), eval labelling,
the LIVE reward (open_r1.live_probe_hint) and the contract test.
"""
from __future__ import annotations
import ast
import os
import re
import torch
import torch.nn as nn

from CPF_utils.chat import user_chat

MC_RE = re.compile(r"<mc>\s*\(?\s*([A-D])", re.IGNORECASE)
MAX_LEN = 2048


def prompt_to_chat(tok, prompt_field) -> str:
    """Accepts the dataset `problem` (plain user text) or the results-jsonl
    `biased_prompt` (str(list of {role, content}) messages, roles human/user/…)."""
    if isinstance(prompt_field, str) and prompt_field.lstrip().startswith("["):
        try:
            prompt_field = ast.literal_eval(prompt_field)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unparseable message list: {prompt_field[:80]!r}") from e
    if isinstance(prompt_field, list):
        msgs = [{"role": ("user" if m.get("role") in ("human", "user", None) else m["role"]),
                 "content": m.get("content", "")} for m in prompt_field]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_chat(tok, str(prompt_field))


def prompt_plain_text(prompt_field) -> str:
    """Last user turn as plain text (what the judge is shown as `question`)."""
    if isinstance(prompt_field, str) and prompt_field.lstrip().startswith("["):
        prompt_field = ast.literal_eval(prompt_field)
    if isinstance(prompt_field, list):
        users = [m.get("content", "") for m in prompt_field if m.get("role") in ("human", "user", None)]
        return users[-1] if users else ""
    return str(prompt_field)


def letter_char_pos(generation: str) -> int | None:
    m = MC_RE.search(generation or "")
    return m.start(1) if m else None


def tok_index_for_char(offsets, char_pos: int) -> int:
    best = -1
    for ti, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s <= char_pos < e:
            return ti
        if s <= char_pos:
            best = ti
    return best


class HintBIntProbe:
    def __init__(self, probe_path: str):
        ck = torch.load(probe_path, map_location="cpu", weights_only=False)
        self.path, self.layer = probe_path, int(ck["layer"])
        self.mean, self.std = ck["mean"].float().reshape(-1), ck["std"].float().reshape(-1)
        self.meta = {k: v for k, v in ck.items() if k not in ("state_dict", "mean", "std")}
        self.probe = nn.Linear(self.mean.shape[0], 2)
        self.probe.load_state_dict(ck["state_dict"])
        self.probe.eval()
        for p in self.probe.parameters():
            p.requires_grad_(False)

    @staticmethod
    @torch.no_grad()
    def cpos_hidden(model, tok, chats: list[str], generations: list[str], layers: list[int],
                    max_length: int = MAX_LEN, pos_offset: int = 0) -> tuple[dict[int, torch.Tensor], list[bool]]:
        """({layer: (B,H) fp32 cpu}, has_cpos[B]). Rows without a C-position get a
        zero vector and has_cpos=False — callers MUST honour the flag.
        Forwarded in chunks of HINT_PROBE_CHUNK rows (default 2): a full GRPO rollout
        batch with eager attention + output_hidden_states OOMs an 80 GB GPU."""
        chunk = int(os.environ.get("HINT_PROBE_CHUNK", 2))
        if len(chats) > chunk:
            hs_parts, ok = [], []
            for s in range(0, len(chats), chunk):
                h, o = HintBIntProbe.cpos_hidden(model, tok, chats[s:s + chunk], generations[s:s + chunk], layers, max_length, pos_offset)
                hs_parts.append(h); ok += o
            return {L: torch.cat([h[L] for h in hs_parts]) for L in layers}, ok
        texts = [c + g for c, g in zip(chats, generations)]
        old = tok.padding_side; tok.padding_side = "right"
        try:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length, return_offsets_mapping=True)
        finally:
            tok.padding_side = old
        offsets = enc.pop("offset_mapping")
        enc = enc.to(model.device)
        idx, ok = [], []
        for b, (c, g) in enumerate(zip(chats, generations)):
            cp = letter_char_pos(g)
            ti = tok_index_for_char(offsets[b].tolist(), len(c) + cp) if cp is not None else -1
            if ti >= 0 and pos_offset:
                ti = max(ti + pos_offset, 0)   # e.g. -1 = B_pre_answer (token before the <mc> letter)
            n = int(enc.attention_mask[b].sum())
            if ti < 0 or ti >= n:          # no <mc> letter, or truncated away
                idx.append(0); ok.append(False)
            else:
                idx.append(ti); ok.append(True)
        was_training = model.training; model.eval()
        try:
            out = model(**enc, output_hidden_states=True, use_cache=False)
        finally:
            if was_training:
                model.train()
        ar = torch.arange(len(texts), device=model.device); it = torch.tensor(idx, device=model.device)
        hs = {L: out.hidden_states[L + 1][ar, it].float().cpu() for L in layers}
        for L in layers:
            hs[L][[i for i, o in enumerate(ok) if not o]] = 0.0
        return hs, ok

    def b_int_from_hidden(self, h: torch.Tensor) -> list[int]:
        return self.probe((h - self.mean) / (self.std + 1e-8)).argmax(-1).tolist()

    def b_int(self, model, tok, prompt_fields, generations) -> list[int | None]:
        """None where the generation has no C-position."""
        chats = [prompt_to_chat(tok, p) for p in prompt_fields]
        hs, ok = self.cpos_hidden(model, tok, chats, generations, [self.layer],
                                  pos_offset=int(self.meta.get("pos_offset", 0)))
        preds = self.b_int_from_hidden(hs[self.layer])
        return [p if o else None for p, o in zip(preds, ok)]
