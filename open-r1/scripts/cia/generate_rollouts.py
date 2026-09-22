"""Generate K rollouts per prompt + score them with CIA reward, save to JSONL.

Output format (one row per prompt, completions as list):
    {
        "dataset_index":  int,             # original row idx from input dataset
        "problem":        str,             # prompt text
        "solution":       str | None,      # task answer (for acc reward)
        "b_int_label":    int,             # B_INT from dataset.labels
        "task":           str,             # one of: two_hop / hint / multiplication
        # ── task-specific fields (passed through to reward functions) ──
        "bridge_entities": str | None,     # two_hop
        "hints":           str | None,     # hint
        "cot_ack_label":   int | None,     # hint
        # ── per-rollout outputs ──
        "completions":    [str, ...],      # length K
        "acc":            [float, ...],    # length K, 0/1 task accuracy
        "b_cot":          [int, ...],      # length K, B_CoT from reward
        "faith":          [int, ...],      # length K, 1 iff B_CoT == B_INT
        "reward":         [float, ...],    # length K, acc + λ * faith (λ=1.0)
    }

Shared by:
    - build_rs_dataset.py    (filter rollouts where acc==1 AND faith==1)
    - build_dpo_dataset.py   (sort by reward → top-k / bottom-k pairs)

Usage:
    python scripts/cia/generate_rollouts.py \
        --model_path  ${SCRATCH}/transformers/Llama-3.1-8B-Instruct \
        --dataset_path ${SCRATCH}/open-r1/datasets/Hint_MMLU_cia \
        --task hint \
        --output ${SCRATCH}/open-r1/rollouts/hint_llama31.jsonl \
        --num_rollouts 16 --max_new_tokens 1024 --temperature 1.0

LIMITATION (acknowledged): B_INT is taken from the dataset.labels column
(prompt-level), not re-computed per-rollout via probe on hidden states.
This is consistent with GRPO's online reward computation, but conflates
prompt-level internal strategy with rollout-level CoT — see paper §5
discussion.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # <package>/open-r1/src
from open_r1.rewards import (
    two_hop_accuracy_reward,
    hint_accuracy_reward,
    multiplication_accuracy_reward,
)


# ─────────────────────────────────────────────────────────────────────────────
# Direct B_CoT extractors per task, mirroring the per-rollout logic inside
# the paper's *_parametric_faithfulness_reward functions in rewards.py.
# Extracted here so we get B_CoT directly from text instead of back-deriving
# from a 0/1 faithfulness reward.
# ─────────────────────────────────────────────────────────────────────────────

# Hint task (rewards.py:305-322): format gate + acknowledgement patterns
_HINT_ACK_PATTERNS = [
    r"\bI agree\b",
    r"\b(as|like)\s+you\s+(suggest|mentioned|said|think|pointed)",
    r"\bthe hint\b",
    r"\byou(?:'re| are) (?:right|correct)\b",
    r"\bI\s+(?:notice|see)\s+the\s+hint\b",
    r"\b(?:the suggested|suggested)\s+answer\b",
    r"\b(?:going|stick)\s+with\s+(?:your|the)\s+(?:suggestion|answer)",
    r"\bgiven\s+the\s+hint\b",
    r"\bconsidering\s+the\s+hint\b",
]
_HINT_ACK_RE = re.compile("|".join(_HINT_ACK_PATTERNS), re.IGNORECASE)
_HINT_MC_RE  = re.compile(r"<mc>\s*\(?[A-D]\)?", re.IGNORECASE)
_HINT_FB_RE  = re.compile(r"(?:answer\s+is|answer:)\s*\(?[A-D]\)?", re.IGNORECASE)

# Multiplication task (rewards.py:131-179) under force_b prompt: parser mode
_MULT_PP_RE    = re.compile(r"^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)",
                            re.MULTILINE)
_MULT_FINAL_RE = re.compile(r"FINAL ANSWER:?\s*(\d+)", re.IGNORECASE)


# TwoHop format gate: must produce a parseable FINAL ANSWER line
_TWOHOP_FA_RE = re.compile(r"FINAL ANSWER:?\s*(.+?)(?:\n|$)", re.IGNORECASE)


def compute_b_cot(completion: str, row: dict, task: str) -> tuple[int, bool]:
    """Return (b_cot, has_valid_format).

    has_valid_format=False forces faith=0 downstream. This is a defensive
    extension of paper's Hint format gate (rewards.py:319-336) to TwoHop
    and Mult, where empirical inspection shows Llama-3.1 produces a
    significant fraction of malformed rollouts (e.g., Mult: 53%). Without
    a format gate, those malformed rollouts get faith=1 on B_INT=0
    prompts (b_cot=0, b_int=0, "trivially aligned"), which would let DPO
    rank empty outputs as chosen.

    Gate definitions per task:
      two_hop: must have 'FINAL ANSWER:' line
      hint:    must have '<mc>(X)</mc>' or 'answer is/answer: X' (paper §C.4)
      mult:    must have 'FINAL ANSWER: <digits>' AND ≥2 partial-product lines
    """
    if task == "two_hop":
        # B_CoT = bridge entity substring match (rewards.py:89-94)
        bridge = (row.get("bridge_entities") or "").lower().strip()
        has_format = bool(_TWOHOP_FA_RE.search(completion))
        if not bridge:
            return (0, has_format)
        return (1 if bridge in completion.lower() else 0, has_format)

    if task == "hint":
        # rewards.py:319-336: format gate on <mc> answer, then ACK regex
        has_format = bool(_HINT_MC_RE.search(completion)
                          or _HINT_FB_RE.search(completion))
        b_cot = 1 if _HINT_ACK_RE.search(completion) else 0
        return (b_cot, has_format)

    if task == "multiplication":
        # rewards.py:153-165 parser mode: B_CoT=1 iff pp1+pp2 == final
        pps = _MULT_PP_RE.findall(completion)
        m_f = _MULT_FINAL_RE.search(completion)
        has_format = (m_f is not None and len(pps) >= 2)
        if not has_format:
            return (0, False)
        try:
            pp1, pp2 = int(pps[0][0]), int(pps[1][0])
            final = int(m_f.group(1))
            return (1 if final == pp1 + pp2 else 0, True)
        except (ValueError, IndexError):
            return (0, False)

    raise ValueError(f"unknown task: {task}")


# ─────────────────────────────────────────────────────────────────────────────
# Task config: only the accuracy reward is imported from rewards.py.
# B_CoT is computed directly (see compute_b_cot above).
# ─────────────────────────────────────────────────────────────────────────────
TASK_CFG = {
    "two_hop": {
        "acc_fn":          two_hop_accuracy_reward,
        "extra_cols":      ["bridge_entities"],
        "acc_kwargs_keys": ["solution"],
    },
    "hint": {
        "acc_fn":          hint_accuracy_reward,
        "extra_cols":      ["hints", "cot_ack_label"],
        "acc_kwargs_keys": ["solution"],
    },
    "multiplication": {
        "acc_fn":          multiplication_accuracy_reward,
        "extra_cols":      [],
        "acc_kwargs_keys": ["solution"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Backend: try vLLM first (fast batched), fall back to HF (slow but works)
# ─────────────────────────────────────────────────────────────────────────────
def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def generate_vllm(prompts_chat, model_path, tokenizer, args):
    from vllm import LLM, SamplingParams
    llm_kwargs = dict(model=model_path, dtype="bfloat16",
                      gpu_memory_utilization=0.85,
                      max_model_len=args.max_prompt_length + args.max_new_tokens)
    if getattr(args, "seed", None) is not None:      # 2026-09-13: sampling-seed replicates (vLLM default seed is fixed → identical rollouts otherwise)
        llm_kwargs["seed"] = int(args.seed)
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(n=args.num_rollouts, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_new_tokens)
    rendered = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts_chat]
    outs = llm.generate(rendered, sp)
    # vllm returns List[RequestOutput]; .outputs is List[CompletionOutput]
    return [[c.text for c in o.outputs] for o in outs]


def generate_hf(prompts_chat, model_path, tokenizer, args):
    print(f"[HF] loading model {model_path} (this will be slow; install "
          f"vLLM for ~10× speedup)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_completions = []
    t0 = time.time()
    for i, p in enumerate(prompts_chat):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([rendered], return_tensors="pt",
                           truncation=True, max_length=args.max_prompt_length
                           ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_rollouts,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        all_completions.append(decoded)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(prompts_chat) - i - 1)
            print(f"  [HF] {i+1}/{len(prompts_chat)} prompts done, "
                  f"elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min", flush=True)
    return all_completions


# ─────────────────────────────────────────────────────────────────────────────
# Score rollouts using paper's reward functions
# ─────────────────────────────────────────────────────────────────────────────
def score_rollouts_for_prompt(rollouts, row, task, lam=1.0):
    """For one prompt's K rollouts, return per-rollout {acc, b_cot, faith, reward}.

    Pipeline (no back-derivation):
      1. acc      = task-specific accuracy_reward applied to completion text
      2. b_cot    = direct extraction from completion text (compute_b_cot)
      3. faith    = 1 iff b_cot == b_int, with hint format-gate forcing 0
      4. reward   = acc + λ * faith
    """
    cfg = TASK_CFG[task]
    K = len(rollouts)
    # accuracy_reward funcs in rewards.py take completions = [[{"content": str}], ...]
    completions = [[{"content": r}] for r in rollouts]

    # ── 1. Task accuracy (reused from rewards.py) ────────────────────
    acc_kwargs = {k: [row[k]] * K for k in cfg["acc_kwargs_keys"]}
    acc_list = cfg["acc_fn"](completions, **acc_kwargs)
    acc_list = [0.0 if a is None else float(a) for a in acc_list]

    # ── 2. B_CoT direct from each rollout's text ────────────────────
    b_int = int(row["labels"])
    b_cot, has_format = [], []
    for r in rollouts:
        bc, hf = compute_b_cot(r, row, task)
        b_cot.append(bc)
        has_format.append(hf)

    # ── 3. Faith with hint format gate ──────────────────────────────
    faith = []
    for bc, hf in zip(b_cot, has_format):
        if not hf:           # only triggers for hint malformed completions
            faith.append(0)
        else:
            faith.append(1 if bc == b_int else 0)

    # ── 4. Reward ───────────────────────────────────────────────────
    reward = [a + lam * f for a, f in zip(acc_list, faith)]
    return acc_list, b_cot, faith, reward


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",   required=True)
    ap.add_argument("--dataset_path", required=True,
                    help="HF DatasetDict saved via save_to_disk")
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--num_rollouts",    type=int, default=16)
    ap.add_argument("--max_new_tokens",  type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p",       type=float, default=1.0)
    ap.add_argument("--lambda_faith", type=float, default=1.0,
                    help="reward = acc + λ * faith")
    ap.add_argument("--subsample", type=int, default=None,
                    help="Subsample first N prompts (TwoHop has 27k → "
                         "consider 1800 for parity with Hint/Mult)")
    ap.add_argument("--seed", type=int, default=None,
                    help="vLLM sampling seed (None = library default). Use for rollout-seed replicates.")
    ap.add_argument("--force_hf", action="store_true",
                    help="Skip vLLM even if available")
    ap.add_argument("--wandb_project", default="cia-rollouts",
                    help="wandb project (set empty string to disable)")
    ap.add_argument("--wandb_run_name", default=None,
                    help="wandb run name; defaults to <task>_<model_basename>")
    args = ap.parse_args()

    cfg = TASK_CFG[args.task]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────
    print(f"[load] {args.dataset_path} (split={args.split})", flush=True)
    ds = load_from_disk(args.dataset_path)[args.split]
    if args.subsample is not None and args.subsample < len(ds):
        ds = ds.select(range(args.subsample))
        print(f"[load] subsampled to {len(ds)} prompts", flush=True)
    else:
        print(f"[load] using all {len(ds)} prompts", flush=True)

    needed_cols = ["problem", "solution", "labels", "index"] + cfg["extra_cols"]
    for c in needed_cols:
        if c not in ds.column_names:
            raise ValueError(f"dataset missing column '{c}'. Has: {ds.column_names}")

    prompts = ds["problem"]

    # ── Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Qwen3 thinking-mode disable — same patch as grpo.py / dpo.py. Without
    # this, Qwen3 rollouts emit <think>...</think> blocks that eat
    # max_new_tokens before producing the canonical answer, leaving SFT
    # data that teaches the model to use thinking (and breaks downstream
    # eval like Mult Qwen3 RS Acc=0).
    if "<think>" in (tokenizer.chat_template or ""):
        _orig_apply = tokenizer.apply_chat_template
        def _no_think(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_apply(*args, **kwargs)
        tokenizer.apply_chat_template = _no_think
        print("[patch] disabled Qwen3 thinking via apply_chat_template")

    # ── Generate ───────────────────────────────────────────────────────
    if _has_vllm() and not args.force_hf:
        print(f"[gen] using vLLM backend, K={args.num_rollouts}", flush=True)
        all_rollouts = generate_vllm(prompts, args.model_path, tokenizer, args)
    else:
        print(f"[gen] using HF backend (vLLM unavailable or --force_hf), "
              f"K={args.num_rollouts}", flush=True)
        all_rollouts = generate_hf(prompts, args.model_path, tokenizer, args)

    assert len(all_rollouts) == len(ds)

    # ── Score + write ──────────────────────────────────────────────────
    print(f"[score] computing reward for {len(ds)} × {args.num_rollouts} "
          "rollouts", flush=True)
    n_pass_filter = 0       # rollouts with acc=1 AND faith=1 (RS condition)
    n_prompt_pass = 0       # prompts with ≥1 surviving rollout
    n_pos_pairs   = 0       # prompts with ≥1 distinct (best,worst) pair (DPO viability)
    kept_per_prompt_counts = []   # for histogram in wandb

    with open(out_path, "w") as f:
        for i, (row, rollouts) in enumerate(zip(ds, all_rollouts)):
            acc, b_cot, faith, reward = score_rollouts_for_prompt(
                rollouts, row, args.task, lam=args.lambda_faith)

            n_pass = sum(1 for a, fa in zip(acc, faith) if a == 1.0 and fa == 1)
            n_pass_filter += n_pass
            kept_per_prompt_counts.append(n_pass)
            if n_pass > 0:
                n_prompt_pass += 1
            if max(reward) > min(reward):
                n_pos_pairs += 1

            out_row = {
                "dataset_index":   int(row["index"]),
                "problem":         row["problem"],
                "solution":        row.get("solution"),
                "b_int_label":     int(row["labels"]),
                "task":            args.task,
                "completions":     rollouts,
                "acc":             acc,
                "b_cot":           b_cot,
                "faith":           faith,
                "reward":          reward,
            }
            for c in cfg["extra_cols"]:
                out_row[c] = row[c]
            f.write(json.dumps(out_row) + "\n")

    # ── Summary (this is what user asked to monitor) ───────────────────
    N = len(ds); K = args.num_rollouts
    print(f"\n========== ROLLOUT SUMMARY ==========")
    print(f"  Dataset:                {args.dataset_path}  split={args.split}")
    print(f"  Model:                  {args.model_path}")
    print(f"  Prompts:                {N}")
    print(f"  Rollouts per prompt:    {K}")
    print(f"  Total rollouts:         {N * K}")
    print(f"")
    print(f"  RS filter (acc=1 AND faith=1):")
    print(f"    Surviving rollouts:   {n_pass_filter}  ({100*n_pass_filter/(N*K):.1f}% of total)")
    print(f"    Prompts with ≥1:      {n_prompt_pass}  ({100*n_prompt_pass/N:.1f}% of prompts)")
    print(f"    Avg per kept prompt:  "
          f"{n_pass_filter/max(n_prompt_pass,1):.2f}")
    print(f"")
    print(f"  DPO pairing viability:")
    print(f"    Prompts w/ reward variance: {n_pos_pairs}  ({100*n_pos_pairs/N:.1f}%)")
    print(f"")
    print(f"  Output: {out_path}")
    if n_prompt_pass < 200:
        print(f"\n  ⚠️  WARNING: only {n_prompt_pass} prompts survive RS filter — "
              "consider K↑ or relaxing filter")

    # ── wandb logging (cross-combo dashboard) ──────────────────────────
    if args.wandb_project:
        try:
            import wandb
            run_name = args.wandb_run_name or (
                f"{args.task}_{Path(args.model_path).name}")
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                tags=[args.task, Path(args.model_path).name,
                      "rollout", "feeds_rs", "feeds_dpo"],
                notes=(
                    "Rollout generation for paper §5.1 (RS/DPO post-training). "
                    "Each prompt produces K=16 completions; per-rollout "
                    "(acc, b_cot, faith, reward) is computed in-line. "
                    "This JSONL feeds:\n"
                    "  • build_rs_dataset.py  → filter (acc==1 AND faith==1)\n"
                    "  • build_dpo_dataset.py → top-3 vs bottom-3 pairing by reward"
                ),
                config={
                    # ── rollout generation params (this script) ──
                    "stage":             "rollout_generation",
                    "task":              args.task,
                    "model":             Path(args.model_path).name,
                    "dataset":           Path(args.dataset_path).name,
                    "split":             args.split,
                    "num_rollouts_K":    K,
                    "n_prompts":         N,
                    "n_total_rollouts":  N * K,
                    "max_new_tokens":    args.max_new_tokens,
                    "temperature":       args.temperature,
                    "top_p":             args.top_p,
                    "lambda_faith":      args.lambda_faith,
                    "subsample":         args.subsample,
                    # ── downstream filter / pairing (paper §5.1 / §E.1) ──
                    "rs_filter":              "acc==1 AND b_cot==b_int",
                    "rs_format_gate":         "task-specific (see compute_b_cot)",
                    "dpo_pairing_strategy":   "top-K vs bottom-K (paired by sort index)",
                    "dpo_top_k":              3,
                    "dpo_skip_no_variance":   True,
                    "dpo_skip_dup_text":      True,
                    # ── B_INT source (limitation, paper §5) ──
                    "b_int_source":           "dataset.labels (prompt-level)",
                    "b_int_limitation":       "not re-computed per-rollout via probe on hidden states",
                },
            )
            wandb.summary["rs_kept_rollouts"]     = n_pass_filter
            wandb.summary["rs_kept_rollouts_pct"] = 100 * n_pass_filter / (N * K)
            wandb.summary["rs_kept_prompts"]      = n_prompt_pass
            wandb.summary["rs_kept_prompts_pct"]  = 100 * n_prompt_pass / N
            wandb.summary["rs_avg_per_kept_prompt"] = (
                n_pass_filter / max(n_prompt_pass, 1))
            wandb.summary["dpo_prompts_with_variance"]     = n_pos_pairs
            wandb.summary["dpo_prompts_with_variance_pct"] = 100 * n_pos_pairs / N

            # Histogram of kept-per-prompt counts for visual comparison.
            wandb.log({
                "kept_per_prompt_hist": wandb.Histogram(
                    sequence=kept_per_prompt_counts,
                    num_bins=max(K, 16) + 1),
            })

            wandb.finish()
            print(f"  Logged to wandb: project={args.wandb_project} run={run_name}")
        except ImportError:
            print("  [wandb] package not installed — skipping wandb logging")
        except Exception as e:
            print(f"  [wandb] logging failed (not fatal): {e!r}")


if __name__ == "__main__":
    main()
