"""Filter scored rollouts → SFT-format HF DatasetDict for Rejection Sampling.

Reads:  rollouts JSONL produced by generate_rollouts.py
Filter: keep rollouts where acc == 1.0 AND faith == 1
        (strict paper §5.1: task-correct AND B_CoT == B_INT)
Output: HF DatasetDict with `train` split (and optional `validation` carry-over)
        in messages format consumable by open-r1/src/open_r1/sft.py.

Output columns:
    messages: [{"role": "user", "content": prompt},
               {"role": "assistant", "content": chosen_completion}]
    dataset_index: int      # for traceability
    task:          str
    reward:        float    # always 2.0 (acc=1, faith=1, λ=1)

Survival stats printed to stdout (this is the "training set size enough?"
check the user asked to monitor).

Usage:
    python scripts/cia/build_rs_dataset.py \
        --rollouts ${SCRATCH}/open-r1/rollouts/hint_llama31.jsonl \
        --out_dir  ${SCRATCH}/open-r1/datasets/Hint_MMLU_rs_llama31

    Optional: --val_rollouts path/to/val_rollouts.jsonl  (carry-over val split)
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def filter_one_file(rows: list[dict], tokenizer=None) -> tuple[list[dict], dict]:
    """Filter to acc=1 AND faith=1. Returns (kept_examples, stats).

    If tokenizer is provided, also renders messages via apply_chat_template
    into a `text` column (required by TRL 0.14 SFTTrainer — it does NOT
    auto-detect messages-format datasets).
    """
    kept = []
    n_total_rollouts = 0
    n_pass = 0
    n_acc_only = 0
    n_faith_only = 0
    per_prompt_kept = []

    for row in rows:
        K = len(row["completions"])
        n_total_rollouts += K
        prompt_kept = 0
        for j in range(K):
            acc = row["acc"][j]
            faith = row["faith"][j]
            if acc == 1.0 and faith == 1:
                messages = [
                    {"role": "user",      "content": row["problem"]},
                    {"role": "assistant", "content": row["completions"][j]},
                ]
                example = {
                    "messages":       messages,
                    "dataset_index":  row["dataset_index"],
                    "task":           row["task"],
                    "reward":         row["reward"][j],
                }
                if tokenizer is not None:
                    example["text"] = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False)
                kept.append(example)
                n_pass += 1
                prompt_kept += 1
            elif acc == 1.0:
                n_acc_only += 1
            elif faith == 1:
                n_faith_only += 1
        per_prompt_kept.append(prompt_kept)

    n_prompts = len(rows)
    n_prompts_with_kept = sum(1 for k in per_prompt_kept if k > 0)
    histogram = Counter(per_prompt_kept)

    stats = {
        "n_prompts":            n_prompts,
        "n_total_rollouts":     n_total_rollouts,
        "n_kept":               n_pass,
        "n_acc_only":           n_acc_only,
        "n_faith_only":         n_faith_only,
        "n_neither":            n_total_rollouts - n_pass - n_acc_only - n_faith_only,
        "n_prompts_with_kept":  n_prompts_with_kept,
        "kept_per_prompt_hist": dict(sorted(histogram.items())),
        "avg_kept_per_kept_prompt":
            n_pass / max(n_prompts_with_kept, 1),
    }
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="train rollouts JSONL")
    ap.add_argument("--val_rollouts", default=None,
                    help="optional val rollouts JSONL (will be filtered same way)")
    ap.add_argument("--out_dir", required=True, help="HF DatasetDict save dir")
    ap.add_argument("--tokenizer_path", default=None,
                    help="If set, render messages → `text` column via apply_chat_template. "
                         "Required for TRL 0.14 SFTTrainer (which has no auto-detect).")
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer_path:
        print(f"Loading tokenizer for chat-template rendering: {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # ── filter train ─────────────────────────────────────────────────
    train_rows = load_jsonl(Path(args.rollouts))
    train_kept, tr_stats = filter_one_file(train_rows, tokenizer=tokenizer)

    print("\n========== RS FILTER STATS (train) ==========")
    print(f"  Input prompts:         {tr_stats['n_prompts']}")
    print(f"  Input rollouts:        {tr_stats['n_total_rollouts']}")
    print(f"  Kept (acc=1, faith=1): {tr_stats['n_kept']}  "
          f"({100*tr_stats['n_kept']/tr_stats['n_total_rollouts']:.2f}%)")
    print(f"  Dropped acc-only:      {tr_stats['n_acc_only']}  "
          f"({100*tr_stats['n_acc_only']/tr_stats['n_total_rollouts']:.2f}%)")
    print(f"  Dropped faith-only:    {tr_stats['n_faith_only']}  "
          f"({100*tr_stats['n_faith_only']/tr_stats['n_total_rollouts']:.2f}%)")
    print(f"  Dropped neither:       {tr_stats['n_neither']}")
    print(f"  Prompts with ≥1 kept:  {tr_stats['n_prompts_with_kept']}  "
          f"({100*tr_stats['n_prompts_with_kept']/tr_stats['n_prompts']:.1f}% "
          "of prompts contribute to SFT)")
    print(f"  Avg kept per kept-prompt: {tr_stats['avg_kept_per_kept_prompt']:.2f}")
    print(f"  Histogram of kept/prompt: {tr_stats['kept_per_prompt_hist']}")

    if tr_stats["n_kept"] < 100:
        print(f"\n  🔴 CRITICAL: only {tr_stats['n_kept']} training examples "
              "survive. SFT will almost certainly not learn anything.")
    elif tr_stats["n_kept"] < 500:
        print(f"\n  🟡 WARNING: only {tr_stats['n_kept']} training examples "
              "survive. SFT may be unstable.")

    splits = {"train": Dataset.from_list(train_kept)}

    # ── filter val (optional) ────────────────────────────────────────
    if args.val_rollouts:
        val_rows = load_jsonl(Path(args.val_rollouts))
        val_kept, val_stats = filter_one_file(val_rows, tokenizer=tokenizer)
        print("\n========== RS FILTER STATS (val) ==========")
        print(f"  Kept: {val_stats['n_kept']} / {val_stats['n_total_rollouts']}")
        splits["validation"] = Dataset.from_list(val_kept)

    # ── save ─────────────────────────────────────────────────────────
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    DatasetDict(splits).save_to_disk(str(out))

    # Save stats next to dataset for traceability
    with open(out / "filter_stats.json", "w") as f:
        json.dump({"train": tr_stats}, f, indent=2)

    print(f"\n  Saved DatasetDict → {out}")
    print(f"  Saved stats       → {out / 'filter_stats.json'}")


if __name__ == "__main__":
    main()
