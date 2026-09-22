"""Live B_INT probe for Multiplication GRPO — thin wrapper around the single
implementation in CPF_utils.mult_bint (probe v2 trained on corruption truth,
last token of chat(full prompt)+pre-summation prefix). No legacy paths, no
short-question / strip-think hacks: the prefix convention is CPF_utils.chat's.
"""
from __future__ import annotations
import os
import sys

_REPO = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from CPF_utils.mult_bint import MultBIntProbe  # noqa: E402


def default_probe_path(model_name: str) -> str:
    d = os.environ.get("MULT_PROBE_DIR", f"{os.environ['SCRATCH']}/results/open-r1/math_results")
    return (f"{d}/2-digit-Multiplication_{model_name}_8888_corruption_FIX_force_b_rdelta9_n3000"
            f"_probe_TRUTH_v2.probe.pt")


class MultLiveProbe:
    def __init__(self, probe_path: str, tokenizer):
        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"Mult probe not found: {probe_path}")
        self.impl = MultBIntProbe(probe_path)
        self.layer = self.impl.layer
        self.position = self.impl.meta.get("probe_position", "pre_summation")
        self.tokenizer = tokenizer
        print(f"[live_probe_mult] probe {probe_path} layer={self.layer} "
              f"trained_with={self.impl.meta.get('trained_with')} test_f1={self.impl.meta.get('test_f1')}", flush=True)

    def compute(self, model, prompts: list[str], completions: list[str]) -> list[int]:
        return self.impl.b_int(model, self.tokenizer, prompts, completions)


_active_probe: MultLiveProbe | None = None
_active_model = None


def set_active(probe, model):
    global _active_probe, _active_model
    _active_probe = probe
    _active_model = model


def get_live_b_int(prompts: list[str], completions: list[str]) -> list[int]:
    if _active_probe is None or _active_model is None:
        raise RuntimeError("Mult LIVE reward called without an active live probe (registration failed?)")
    return _active_probe.compute(_active_model, prompts, completions)
