"""Per-model / per-task probe layer configuration from paper Table 7.

Usage:
    from CPF_utils.layer_config import get_best_layer
    layer = get_best_layer(model_name, task)   # raises if missing

Keys are the HuggingFace model directory name (last path component) and one
of {'two_hop', 'hint', 'multiplication'}.
"""
from __future__ import annotations

# (model_name, task) -> layer index (0-indexed). Mult layers updated 2026-05-06
# to match the actual force_b_rdelta9 gridsearch JSONs in
# ${SCRATCH}/results/open-r1/math_results/probe_gridsearch/. Llama-3.1
# entries added (previously fell back to n_layers/2 = 16 for everything).
BEST_LAYER: dict[tuple[str, str], int] = {
    # Llama-3.1-8B-Instruct (PRIMARY model post-2026-05-01)
    ("Llama-3.1-8B-Instruct", "two_hop"):        16,
    ("Llama-3.1-8B-Instruct", "hint"):           18,
    ("Llama-3.1-8B-Instruct", "multiplication"): 21,   # gridsearch
    # Meta-Llama-3-8B-Instruct (legacy, kept for the Meta-Llama-3 ablation)
    ("Meta-Llama-3-8B-Instruct", "two_hop"):        16,
    ("Meta-Llama-3-8B-Instruct", "hint"):           18,
    ("Meta-Llama-3-8B-Instruct", "multiplication"): 14,
    # Gemma-2-9B-it
    ("gemma-2-9b-it", "two_hop"):        22,
    ("gemma-2-9b-it", "hint"):           24,
    ("gemma-2-9b-it", "multiplication"): 27,   # gridsearch (was 20)
    # Qwen3-8B
    ("Qwen3-8B", "two_hop"):        18,
    ("Qwen3-8B", "hint"):           16,
    ("Qwen3-8B", "multiplication"): 21,   # gridsearch (was 15)
}

# Linear probe hyperparameters. Mult entries updated 2026-05-06 to match
# the gridsearch / hparam-sweep JSONs (see Table 6 in the paper).
PROBE_HPARAMS: dict[tuple[str, str], dict] = {
    ("Llama-3.1-8B-Instruct", "two_hop"):        dict(epochs=15, lr=1e-3, weight_decay=0.01),
    ("Llama-3.1-8B-Instruct", "hint"):           dict(epochs=10, lr=1e-3, weight_decay=0.01),
    ("Llama-3.1-8B-Instruct", "multiplication"): dict(epochs=25, lr=1e-4, weight_decay=0.0),
    ("Meta-Llama-3-8B-Instruct", "two_hop"):        dict(epochs=15, lr=1e-3, weight_decay=0.01),
    ("Meta-Llama-3-8B-Instruct", "hint"):           dict(epochs=10, lr=1e-3, weight_decay=0.01),
    ("Meta-Llama-3-8B-Instruct", "multiplication"): dict(epochs=12, lr=5e-4, weight_decay=0.01),
    ("gemma-2-9b-it", "two_hop"):        dict(epochs=15, lr=1e-3, weight_decay=0.01),
    ("gemma-2-9b-it", "hint"):           dict(epochs=10, lr=1e-3, weight_decay=0.01),
    ("gemma-2-9b-it", "multiplication"): dict(epochs=25, lr=5e-4, weight_decay=0.10),
    ("Qwen3-8B", "two_hop"):        dict(epochs=12, lr=2e-3, weight_decay=0.01),
    ("Qwen3-8B", "hint"):           dict(epochs=10, lr=1e-3, weight_decay=0.01),
    ("Qwen3-8B", "multiplication"): dict(epochs=25, lr=1e-4, weight_decay=0.0),
}

# Task name map from dataset_name → canonical task key.
DATASET_TO_TASK = {
    "TwoHopFact":    "two_hop",
    "HoppingtooLate": "two_hop",
    "SOCRATES":      "two_hop",
    "Hint_MMLU":     "hint",
    "Hint_GPQA":     "hint",
    "2-digit-Multiplication":   "multiplication",
    "3-digit-Multiplication":   "multiplication",
    "4-digit-Multiplication":   "multiplication",
    "Multiplication":           "multiplication",
}


def _normalize_model(model_name: str) -> str:
    return model_name.split("/")[-1]


def resolve_task(dataset_name: str) -> str:
    if dataset_name in DATASET_TO_TASK:
        return DATASET_TO_TASK[dataset_name]
    # Fuzzy fallback for dataset variants that embed the task key.
    low = dataset_name.lower()
    for k, v in DATASET_TO_TASK.items():
        if k.lower() in low:
            return v
    raise KeyError(f"Cannot resolve task for dataset: {dataset_name}")


def get_best_layer(model_name: str, task_or_dataset: str,
                   n_layers: int | None = None, strict: bool = False) -> int:
    """Return best-layer index per Table 7. If not found, fall back to the
    middle of the model when `strict=False`.
    """
    model_key = _normalize_model(model_name)
    # Accept either 'two_hop' etc. or raw dataset name.
    task = task_or_dataset if task_or_dataset in {
        "two_hop", "hint", "multiplication"} else resolve_task(task_or_dataset)
    key = (model_key, task)
    if key in BEST_LAYER:
        return BEST_LAYER[key]
    if strict:
        raise KeyError(f"No best-layer entry for {key}")
    if n_layers is None:
        raise ValueError(
            f"No entry for {key} and n_layers not provided for fallback")
    return n_layers // 2


def get_probe_hparams(model_name: str, task_or_dataset: str) -> dict:
    model_key = _normalize_model(model_name)
    task = task_or_dataset if task_or_dataset in {
        "two_hop", "hint", "multiplication"} else resolve_task(task_or_dataset)
    return PROBE_HPARAMS.get(
        (model_key, task),
        dict(epochs=15, lr=1e-3, weight_decay=0.01),
    )
