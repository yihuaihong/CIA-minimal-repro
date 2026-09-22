"""Pair scored rollouts → DPO-format HF DatasetDict.

Pairing strategy (user decision 2026-05-17): top-3 vs bottom-3 within each
prompt's G=16 rollouts, yielding 3 preference pairs per prompt.

Reads:  rollouts JSONL produced by generate_rollouts.py
Output: HF DatasetDict consumable by open-r1/src/open_r1/dpo.py
        Columns follow TRL DPOTrainer's standard "implicit" format:
            prompt           : str
            chosen           : str   (winner completion only, no prompt)
            rejected         : str
            chosen_reward    : float  (for traceability, NOT used by trainer)
            rejected_reward  : float
            dataset_index    : int
            task             : str

Pairing rules:
  - Within each prompt:
      * Sort G rollouts by reward descending.
      * Take top K=3 as candidates for `chosen`, bottom K=3 as `rejected`.
      * Form 3 pairs by index (top-i with bottom-i).
  - Skip prompts where top-K reward == bottom-K reward (no preference signal).
  - Skip prompts where chosen and rejected completions are textually identical.

Usage:
    python scripts/cia/build_dpo_dataset.py \
        --rollouts ${SCRATCH}/open-r1/rollouts/hint_llama31.jsonl \
        --out_dir  ${SCRATCH}/open-r1/datasets/Hint_MMLU_dpo_llama31 \
        --top_k 3
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_pairs_one_file(rows: list[dict], top_k: int) -> tuple[list[dict], dict]:
    pairs = []
    n_prompts_used    = 0
    n_dropped_no_var  = 0
    n_dropped_dup     = 0
    n_pairs_per_prompt = []

    for row in rows:
        K = len(row["completions"])
        if K < 2 * top_k:
            # not enough rollouts to form top-k vs bottom-k
            continue
        rewards = np.array(row["reward"])
        order = np.argsort(-rewards)         # descending
        top_idx = order[:top_k]
        bot_idx = order[-top_k:]

        # filter pairs: must have reward differential AND non-identical text
        prompt_pairs = []
        for ci, ri in zip(top_idx, bot_idx):
            if rewards[ci] <= rewards[ri]:
                continue   # no preference signal for this pair
            chosen_text   = row["completions"][int(ci)]
            rejected_text = row["completions"][int(ri)]
            if chosen_text.strip() == rejected_text.strip():
                continue
            prompt_pairs.append({
                "prompt":           row["problem"],
                "chosen":           chosen_text,
                "rejected":         rejected_text,
                "chosen_reward":    float(rewards[ci]),
                "rejected_reward":  float(rewards[ri]),
                "dataset_index":    row["dataset_index"],
                "task":             row["task"],
            })

        if len(prompt_pairs) == 0:
            if rewards.max() == rewards.min():
                n_dropped_no_var += 1
            else:
                n_dropped_dup += 1
            n_pairs_per_prompt.append(0)
        else:
            pairs.extend(prompt_pairs)
            n_prompts_used += 1
            n_pairs_per_prompt.append(len(prompt_pairs))

    histogram = Counter(n_pairs_per_prompt)
    stats = {
        "n_prompts":         len(rows),
        "n_prompts_used":    n_prompts_used,
        "n_dropped_no_var":  n_dropped_no_var,
        "n_dropped_all_dup": n_dropped_dup,
        "n_pairs_total":     len(pairs),
        "pairs_per_prompt_hist": dict(sorted(histogram.items())),
        "avg_pairs_per_used_prompt":
            len(pairs) / max(n_prompts_used, 1),
    }
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--val_rollouts", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=3,
                    help="form top-K vs bottom-K pairs per prompt")
    args = ap.parse_args()

    train_rows = load_jsonl(Path(args.rollouts))
    train_pairs, tr_stats = build_pairs_one_file(train_rows, args.top_k)

    print("\n========== DPO PAIRING STATS (train) ==========")
    print(f"  Input prompts:                   {tr_stats['n_prompts']}")
    print(f"  Prompts contributing pairs:      {tr_stats['n_prompts_used']}  "
          f"({100*tr_stats['n_prompts_used']/tr_stats['n_prompts']:.1f}%)")
    print(f"  Dropped (no reward variance):    {tr_stats['n_dropped_no_var']}")
    print(f"  Dropped (all candidate pairs identical text): "
          f"{tr_stats['n_dropped_all_dup']}")
    print(f"  Total preference pairs:          {tr_stats['n_pairs_total']}")
    print(f"  Avg pairs per used prompt:       "
          f"{tr_stats['avg_pairs_per_used_prompt']:.2f}")
    print(f"  Pairs/prompt histogram:          {tr_stats['pairs_per_prompt_hist']}")

    if tr_stats["n_pairs_total"] < 200:
        print(f"\n  🔴 CRITICAL: only {tr_stats['n_pairs_total']} DPO pairs — "
              "DPO will likely overfit or stall.")
    elif tr_stats["n_pairs_total"] < 1000:
        print(f"\n  🟡 WARNING: only {tr_stats['n_pairs_total']} DPO pairs — "
              "may be insufficient for strong signal.")

    splits = {"train": Dataset.from_list(train_pairs)}

    if args.val_rollouts:
        val_rows = load_jsonl(Path(args.val_rollouts))
        val_pairs, val_stats = build_pairs_one_file(val_rows, args.top_k)
        print(f"\n  Val pairs: {val_stats['n_pairs_total']} from "
              f"{val_stats['n_prompts_used']} prompts")
        splits["validation"] = Dataset.from_list(val_pairs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    DatasetDict(splits).save_to_disk(str(out))

    with open(out / "pairing_stats.json", "w") as f:
        json.dump({"train": tr_stats, "top_k": args.top_k}, f, indent=2)

    print(f"\n  Saved DatasetDict → {out}")
    print(f"  Saved stats       → {out / 'pairing_stats.json'}")


if __name__ == "__main__":
    main()
