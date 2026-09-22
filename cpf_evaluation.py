import os
import torch
import random
import numpy as np
from tqdm import tqdm
from os.path import join
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from CPF_utils import data_utils, evaluation_utils


# ======================
# Argument parsing
# ======================

def parse_args():
    parser = argparse.ArgumentParser()

    # reproducibility
    parser.add_argument("--seed", type=int, default=8888) #5555 6666 7777 8888 9999
    parser.add_argument("--device", type=int, default=0)

    # model
    parser.add_argument("--model_dir", type=str, default="${SCRATCH}/transformers")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b-it")

    # dataset
    parser.add_argument("--dataset_name", type=str, default="TwoHopFact")
    parser.add_argument("--dataset_dir", type=str, default="${SCRATCH}/datasets")

    # evaluation switches
    parser.add_argument("--eval_acc", action="store_true")
    parser.add_argument("--eval_cpf", action="store_true")

    # eval hyperparams
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--use_cot_prompt",
        action="store_true",  # 加这个参数就 True，不加就 False
        help="是否使用竖式 CoT 提示（默认不使用，即 direct 模式）"
    )

    # datasets hyperparams 调试用，后面删掉，用于从数据集中抽样，用于快速跑通数据集
    parser.add_argument("--sample_num", type=int, default=0)

    # paper-rigor: restrict eval to test-split indices from a prepped HF
    # DatasetDict (paths like
    # ${SCRATCH}/open-r1/datasets/<TwoHopFact_cia_*|Hint_MMLU_cia*|Mult2d_cia_*>).
    parser.add_argument("--test_indices_from", type=str, default=None,
                        help="Path to HF DatasetDict; restrict eval to its 'test' split indices.")
    parser.add_argument("--indices_split", type=str, default="test",
                        help="which split of --test_indices_from to use (test | validation); VAL is used for checkpoint selection")

    return parser.parse_args()


# ======================
# Main
# ======================

def main():
    args = parse_args()

    # ----- seed -----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tqdm.pandas()

    # ----- device -----
    torch.cuda.set_device(args.device)

    # ----- model & tokenizer -----
    model = AutoModelForCausalLM.from_pretrained(
        join(args.model_dir, args.model_name),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        join(args.model_dir, args.model_name),
        trust_remote_code=True
    )

    if "qwen" in model.config.model_type.lower():
        tokenizer.pad_token = "<|endoftext|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Note: PROBE_BASE_MODEL env var (read inside probing_utils) overrides the
    # probe-cache lookup key only. Filenames still use this run's name. We do
    # NOT mutate model.config._name_or_path here — that would break the
    # filename suffix used for outputs (labeled jsonl, etc.).

    num_layers = model.config.num_hidden_layers
    print(f"Loaded model with {num_layers} layers")

    # ----- dataset -----
    dataset = data_utils.load_dataset(
        dataset_name = args.dataset_name,
        dataset_dir = args.dataset_dir,
        sample_num = args.sample_num,
        seed = args.seed
    )

    # Filter to test-split indices from a prepped HF DatasetDict if provided.
    # Used for paper-rigor eval: only run on samples the model never saw
    # during training (matches the 9119/600 test splits in prep_*_grpo.py).
    if args.test_indices_from:
        import pandas as pd
        from datasets import load_from_disk
        prepped = load_from_disk(args.test_indices_from)
        if args.indices_split not in prepped:
            raise ValueError(f"No '{args.indices_split}' split in {args.test_indices_from}")
        test_idx = set(int(x) for x in prepped[args.indices_split]["index"])
        if isinstance(dataset, pd.DataFrame):
            # TwoHopFact-style DataFrame: filter by `index` column or by
            # DataFrame integer index. The CSV order matches original index.
            if "index" in dataset.columns:
                mask = dataset["index"].astype(int).isin(test_idx)
                n_before = len(dataset)
                dataset = dataset[mask].reset_index(drop=True)
                print(f"[test-split] {n_before} → {len(dataset)} samples (by 'index' col)")
            else:
                # Use df position as the index used by prep_two_hop_grpo
                n_before = len(dataset)
                dataset = dataset.loc[dataset.index.isin(test_idx)].reset_index(drop=True)
                print(f"[test-split] {n_before} → {len(dataset)} samples (by df position)")
        elif isinstance(dataset, list):
            # List-type datasets (Hint, Mult). Different prep scripts apply
            # different pre-filters, so reproduce per task.
            if args.dataset_name == "Hint_MMLU":
                if os.environ.get("HINT_FULL_SET"):
                    # FULL set (2026-05-31): keep BOTH suggestion_False
                    # (single-turn) and posthoc_False (multi-turn). The default
                    # branch below filtered to single-turn only.
                    pre = list(dataset)
                    pre_label = "full(posthoc+suggestion)"
                else:
                    # prep_hint_grpo drops posthoc multi-turn samples then
                    # enumerates the filtered list.
                    pre = [
                        r for r in dataset
                        if isinstance(r.get("biased_prompt"), list)
                        and len(r["biased_prompt"]) == 1
                    ]
                    pre_label = "single_turn"
            else:
                # prep_multiplication_grpo enumerates the loaded list directly
                # (no extra filter).
                pre = list(dataset)
                pre_label = "raw"
            enumerated = [{**r, "index": i} for i, r in enumerate(pre)]
            if args.dataset_name == "Hint_MMLU":
                # Index spaces differ between the source records and the prepped datasets
                # (parse-fail filtering, per-model datasets), so restrict by PROMPT TEXT:
                # keep records whose single-turn biased prompt equals a `problem` of the split.
                problems = set(p.strip() for p in prepped[args.indices_split]["problem"])
                def _txt(r):
                    bp = r.get("biased_prompt")
                    if isinstance(bp, list) and bp:
                        return str(bp[-1].get("content", "")).strip()
                    return str(bp).strip()
                kept = [r for r in enumerated if _txt(r) in problems]
            else:
                kept = [r for r in enumerated if r["index"] in test_idx]
            print(f"[test-split] {len(dataset)} raw → {len(pre)} {pre_label} → "
                  f"{len(kept)} test-split")
            dataset = kept
        else:
            print(f"[test-split] WARNING unsupported type {type(dataset)}; skip filter")

    # ----- evaluation -----
    if args.eval_acc:
        acc = evaluation_utils.accuracy_evaluation(
            model,
            args.model_name,
            dataset,
            args.dataset_name,
            tokenizer,
            batch_size=args.batch_size,
            use_cot_prompt=args.use_cot_prompt,
            seed=args.seed,
        )
        # print(f"accuracy_results: {acc} on Dataset: {args.dataset_name} with sample_num: {args.sample_num}")

    if args.eval_cpf:
        cpf = evaluation_utils.CPF_evaluation(
            model,
            args.model_name,
            dataset,
            args.dataset_name,
            tokenizer,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"CPF_results: {cpf} on Dataset: {args.dataset_name} with sample_num: {args.sample_num}")


if __name__ == "__main__":
    main()

# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name TwoHopFact
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 64 --sample_num 500 --dataset_name TwoHopFact
# python cpf_evaluation.py --eval_cpf --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name TwoHopFact
# python cpf_evaluation.py --eval_cpf --use_cot_prompt --batch_size 32 --dataset_name TwoHopFact

# Example Command: python cpf_evaluation.py --eval_cpf --use_cot_prompt --batch_size 64 --sample_num 100 --dataset_name SOCRATES
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 64 --sample_num 0 --dataset_name Hint_MMLU
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name 2-digit-Multiplication
# python cpf_evaluation.py --eval_acc --batch_size 32 --sample_num 500 --dataset_name 2-digit-Multiplication
# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name 2-digit-Multiplication --model_name Meta-Llama-3-8B-Instruct #Qwen3-8B

# python cpf_evaluation.py --eval_acc --use_cot_prompt --batch_size 32 --sample_num 500 --dataset_name Hint_MMLU