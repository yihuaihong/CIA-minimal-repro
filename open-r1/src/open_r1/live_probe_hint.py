"""Live B_INT probe for Hint GRPO — thin wrapper around CPF_utils.hint_bint
(C-position probe v2). Rows whose completion has no `<mc>` letter get B_INT=None
(format failure); no last-token fallback.
"""
from __future__ import annotations
import os
import sys

_REPO = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from CPF_utils.hint_bint import HintBIntProbe  # noqa: E402


def hint_probe_path(model_name: str) -> str:
    if os.environ.get("HINT_PROBE_FILE"):          # reward-side override (e.g. B-position pilot);
        return os.environ["HINT_PROBE_FILE"]       # the EVAL probe stays C-pos v2 regardless.
    d = os.environ.get("HINT_PROBE_DIR", f"{os.environ['SCRATCH']}/results/open-r1/hint_mmlu_results")
    return f"{d}/hint_{model_name}_cpos_probe_v2.pt"


class HintLiveProbe:
    def __init__(self, probe_pt_path: str, tokenizer):
        if not os.path.exists(probe_pt_path):
            raise FileNotFoundError(f"Hint C-pos probe v2 missing: {probe_pt_path}")
        self.impl = HintBIntProbe(probe_pt_path)
        # optional second probe (HINT_PROBE_FILE2): ensemble by summed standardized logits.
        self.impl2 = HintBIntProbe(os.environ["HINT_PROBE_FILE2"]) if os.environ.get("HINT_PROBE_FILE2") else None
        self.layer = self.impl.layer
        self.position = self.impl.meta.get("position", "C_mc_letter")
        self.tokenizer = tokenizer
        print(f"[live_probe_hint] probe {probe_pt_path} layer={self.layer} "
              f"trained_with={self.impl.meta.get('trained_with')} test_f1={self.impl.meta.get('test_f1')}", flush=True)

    def compute(self, model, prompts: list[str], completions: list[str]) -> list[int | None]:
        if self.impl2 is None:
            return self.impl.b_int(model, self.tokenizer, prompts, completions)
        from CPF_utils.hint_bint import prompt_to_chat
        chats = [prompt_to_chat(self.tokenizer, p) for p in prompts]
        outs = []
        for imp in (self.impl, self.impl2):
            hs, ok = imp.cpos_hidden(model, self.tokenizer, chats, completions, [imp.layer],
                                     pos_offset=int(imp.meta.get("pos_offset", 0)))
            z = (hs[imp.layer] - imp.mean) / (imp.std + 1e-8)
            outs.append((imp.probe(z), ok))
        logits = outs[0][0] + outs[1][0]
        ok = [a and b for a, b in zip(outs[0][1], outs[1][1])]
        preds = logits.argmax(-1).tolist()
        return [p if o else None for p, o in zip(preds, ok)]


_active_probe: HintLiveProbe | None = None
_active_model = None


def set_active(probe, model):
    global _active_probe, _active_model
    _active_probe = probe
    _active_model = model


def get_live_b_int(prompts: list[str], completions: list[str]) -> list[int | None]:
    if _active_probe is None or _active_model is None:
        raise RuntimeError("Hint LIVE reward called without an active live probe (registration failed?)")
    return _active_probe.compute(_active_model, prompts, completions)
