"""SINGLE SOURCE OF TRUTH for the Multiplication B_INT probe read-out and B_CoT.

Definition (setup S-MULT-PROBE-v2 / S-MULT-EVAL-v2, user decision 2026-08-25):
  B_INT = argmax of a frozen linear probe on the hidden state at the LAST token of
          chat(full force-B prompt) + CoT prefix up to (not incl.) the summation
          line (`locate_pre_summation_prefix`), at the probe's layer, features
          standardized with the probe's train-set mean/std.
          The probe is trained on corruption GROUND-TRUTH labels
          (`follows_partial_products`, train split) — not distilled from an older probe.
  B_CoT = the written CoT is arithmetically self-consistent: final == pp1 + pp2.
Reward (LIVE) and eval both call this module. Qwen3 prefixes always carry the
empty think block (CPF_utils.chat convention).
"""
from __future__ import annotations
import re
import torch
import torch.nn as nn

from CPF_utils.mult_probe import locate_pre_summation_prefix, _PP_RE, _FINAL_RE  # single regex source
from CPF_utils.chat import user_chat

MAX_LEN = 1024


def b_cot_selfcon(text: str) -> int:
    pps = _PP_RE.findall(text or "")
    m = _FINAL_RE.search(text or "")
    if len(pps) < 2 or m is None:
        return 0
    return int(int(m.group(1)) == int(pps[0][0]) + int(pps[1][0]))


def approach(text: str) -> str:
    """'B' if the CoT contains ≥2 partial-product lines (long multiplication), else 'A'."""
    return "B" if len(_PP_RE.findall(text or "")) >= 2 else "A"


def probe_text(tok, full_prompt: str, completion: str) -> str:
    return user_chat(tok, full_prompt) + (locate_pre_summation_prefix(completion) or "")


class MultBIntProbe:
    def __init__(self, probe_path: str):
        ck = torch.load(probe_path, map_location="cpu", weights_only=False)
        self.path, self.layer = probe_path, int(ck["layer"])
        self.mean, self.std = ck["mean"].float(), ck["std"].float()
        self.meta = {k: v for k, v in ck.items() if k not in ("state_dict", "mean", "std")}
        self.probe = nn.Linear(self.mean.shape[0], 2)
        self.probe.load_state_dict(ck["state_dict"])
        self.probe.eval()
        for p in self.probe.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def last_hidden(self, model, tok, texts: list[str], layers: list[int] | None = None,
                    max_length: int = MAX_LEN) -> dict[int, torch.Tensor]:
        """{layer: (B,H) fp32 cpu} hidden states at the last real token.
        Rows are forwarded ONE AT A TIME (no padding): with bf16 kernels a padded batch
        flips the argmax of borderline rows (16% for Qwen3 between batch 8 and 64),
        which would make the reward depend on batch composition."""
        if len(texts) > 1:
            parts = [self.last_hidden(model, tok, [t], layers, max_length) for t in texts]
            return {L: torch.cat([p[L] for p in parts]) for L in (layers or [self.layer])}
        old = tok.padding_side; tok.padding_side = "right"
        try:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(model.device)
        finally:
            tok.padding_side = old
        was_training = model.training; model.eval()
        try:
            out = model(**enc, output_hidden_states=True, use_cache=False)
        finally:
            if was_training:
                model.train()
        last = enc.attention_mask.sum(1) - 1
        ar = torch.arange(len(texts), device=last.device)
        layers = layers or [self.layer]
        return {L: out.hidden_states[L + 1][ar, last].float().cpu() for L in layers}

    def b_int_from_hidden(self, h: torch.Tensor) -> list[int]:
        z = (h - self.mean) / (self.std + 1e-8)
        return self.probe(z).argmax(-1).tolist()

    def b_int(self, model, tok, full_prompts: list[str], completions: list[str]) -> list[int]:
        texts = [probe_text(tok, p, c) for p, c in zip(full_prompts, completions)]
        return self.b_int_from_hidden(self.last_hidden(model, tok, texts)[self.layer])

    def cells(self, model, tok, full_prompts, completions) -> list[tuple[int, int]]:
        bi = self.b_int(model, tok, full_prompts, completions)
        return [(b, b_cot_selfcon(c)) for b, c in zip(bi, completions)]
