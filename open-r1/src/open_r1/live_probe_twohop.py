"""Live B_INT probe for TwoHop GRPO training — thin wrapper around the single
implementation in CPF_utils.twohop_bint (2-position union, p1 = last e1 token
in the prompt, p2 = last e1 token after CoT step 1, top-K with K from
TWOHOP_TOP_K). No fallbacks: inner_subjects and completions are REQUIRED.
"""
from __future__ import annotations
import os
import sys

_REPO = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from CPF_utils.twohop_bint import TwoHopBIntProbe, default_probe_path, TOP_K_DEFAULT  # noqa: E402


class TwoHopLiveProbe:
    TOP_K = TOP_K_DEFAULT

    def __init__(self, model_name: str, tokenizer, probe_path: str | None = None):
        probe_path = probe_path or default_probe_path(model_name)
        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"TwoHop probe not found for {model_name}: {probe_path}")
        self.impl = TwoHopBIntProbe(probe_path)
        self.layer = self.impl.layer
        self.tokenizer = tokenizer
        print(f"[live_probe_twohop] probe {probe_path} layer={self.layer} K={self.TOP_K} "
              f"hp={self.impl.hp}", flush=True)

    def compute(self, model, prompts: list[str], correct_bridges: list[str],
                inner_subjects: list[str], completions: list[str] | None = None,
                max_length: int = 768) -> list[int]:
        """B_INT(gold bridge) per row. completions=None → prompt-level (p1 only)."""
        return self.impl.b_int(model, self.tokenizer, prompts, inner_subjects,
                               correct_bridges, completions=completions, k=self.TOP_K)


_active_probe: TwoHopLiveProbe | None = None
_active_model = None


def set_active(probe, model):
    global _active_probe, _active_model
    _active_probe = probe
    _active_model = model


def get_live_b_int(prompts: list[str], correct_bridges: list[str],
                   inner_subjects: list[str] | None = None,
                   completions: list[str] | None = None) -> list[int] | None:
    if _active_probe is None or _active_model is None:
        return None
    if inner_subjects is None or completions is None:
        raise ValueError("TwoHop LIVE B_INT requires inner_subjects and completions (2-position union)")
    return _active_probe.compute(_active_model, prompts, correct_bridges,
                                 inner_subjects=inner_subjects, completions=completions)
