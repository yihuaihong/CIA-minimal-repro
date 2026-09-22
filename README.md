# CIA: minimal code for RS-B, DPO-A and evaluation

This repository contains the code for two training methods, **RS-B** and **DPO-A**, and their evaluation on three tasks: **TwoHop**, **Hint** and **Mult**. It is the minimal subset of a larger research codebase needed to run these pipelines, extracted by following the actual import graph of the entry scripts. It contains code only: no data, model weights, probe weights or experimental results.

A Chinese version of this README is in [README_zh.md](README_zh.md).

## 1. What is measured

Each model response gets two binary labels:

- **B_INT**: whether the model internally used the key information, judged by a linear probe or by a causal intervention.
- **B_CoT**: whether the chain of thought (CoT) states it.

**CIA** is the macro-F1 agreement between B_INT and B_CoT. We write faith = 1[B_INT == B_CoT], and acc for answer correctness. Training labels and evaluation labels use the same definitions.

| Task | B_INT | B_CoT |
|---|---|---|
| TwoHop | A linear probe reads the bridge entity at the last e1 token (K = 100) | Whether the CoT names the bridge entity |
| Hint | A probe at the answer-letter position reads whether the hint was used | An LLM judge (Qwen2.5-32B-Instruct) decides whether the CoT acknowledges the hint |
| Mult | Training labels: a probe reads the partial products before summation. Primary evaluation metric: a causal corruption test that alters the written partial products and checks whether the final answer follows | Whether the final answer equals pp1 + pp2 (self-consistency) |

## 2. Training methods

Both methods start from the same data. The base model samples 16 responses per training prompt (T = 1.0, top-p 1.0), and each response is labelled as above.

| Method | Training data | Objective | Hyperparameters |
|---|---|---|---|
| **RS-B** | Keep every response with faith = 1, whether or not it is correct | SFT (`open-r1/src/open_r1/sft.py`) | lr 1e-6, cosine, warmup 0.1, 1 or 2 epochs (see each recipe) |
| **DPO-A** | Per prompt, rank responses by acc + faith and pair the top 3 against the bottom 3 | DPO (`open-r1/src/open_r1/dpo.py`) | β 0.1, lr 5e-7, 1 epoch, sigmoid loss |

- TwoHop recipes save a checkpoint every 15 steps, and evaluation selects the peak on the validation split. Hint and Mult recipes save only the final model.
- Recipes for every task and model are in `open-r1/recipes/CIA/rs/` (RS-B) and `open-r1/recipes/CIA/dpo/` (DPO-A).
- Training uses accelerate with DeepSpeed ZeRO-3, configured in `open-r1/recipes/accelerate_configs/zero3_no_offload_tuned.yaml`.

## 3. Layout

```
run/                     entry points (start here)
  env.sh                 shared environment
  train_offline.sh       rollouts → labels → RS-B / DPO-A datasets → training
  eval_twohop.sh         TwoHop evaluation
  eval_hint.sh           Hint evaluation
  eval_mult.sh           Mult evaluation, step 1: generation + probe-based CIA (secondary)
  eval_mult_causal.sh    Mult evaluation, step 2: causal corruption metric (primary)
  paired_bootstrap.sh    paired bootstrap of a trained model against the base model
open-r1/                 training framework (fork of HuggingFace open-r1, Apache-2.0, see open-r1/LICENSE)
  scripts/cia/           generate_rollouts.py, build_rs_dataset.py, build_dpo_dataset.py
  src/open_r1/           sft.py, dpo.py and their dependencies
  recipes/               RS-B / DPO-A yaml files and the accelerate config
scripts/                 labelling, evaluation and bootstrap scripts
CPF_utils/               probes, B_INT / B_CoT definitions, metrics, corruption test
cpf_evaluation.py        generation entry point for Hint evaluation
```

## 4. Installation

```bash
conda create -n cia python=3.11 -y && conda activate cia
pip install -r requirements.txt
```

**GPUs:**
- Training uses 2 GPUs by default (`NUM_GPUS`). The original experiments ran on 80 GB A100, H100 and H200 GPUs.
- Labelling Hint rollouts needs one more GPU for the judge server (Qwen2.5-32B-Instruct), so 3 in total.
- Hint evaluation loads the judge in-process and needs a GPU that fits a 32B model in bf16.

**wandb:** recipes set `report_to: wandb`. Set `WANDB_MODE=offline`, or change `report_to` to `none` in the recipes, if you do not use it.

## 5. Inputs you must provide

These are not in the repository. All paths are relative to the environment variable `$SCRATCH`.

| Input | Path |
|---|---|
| Base model weights | `transformers/{Qwen3-8B, gemma-2-9b-it, Llama-3.1-8B-Instruct}` |
| Hint judge model | `transformers/Qwen2.5-32B-Instruct` |
| TwoHop task data (HF DatasetDict) | `open-r1/datasets/TwoHopFact_cia_<MODEL>_linear_probe_v3_with_inner_subj` |
| TwoHop raw CSV (evaluation) | `datasets/TwoHopFact/TwoHopFact.csv` |
| TwoHop probe | `results/open-r1/probing_results/probe_chat_filtered_trainsplit_sp/probe_chat_filtered_<MODEL>.pt` |
| Hint task data (HF DatasetDict) | `open-r1/datasets/Hint_MMLU_cia` |
| Hint raw data (evaluation) | the files under `datasets/` that `cpf_evaluation.py --dataset_name Hint_MMLU` reads |
| Hint probe | `results/open-r1/hint_mmlu_results/hint_<MODEL>_cpos_probe_v2.pt` |
| Mult prompt data (HF DatasetDict) | `open-r1/datasets/Mult2d_cia_<MODEL>_force_b_rdelta9_strict` |
| Mult base-model corruption file | `results/open-r1/math_results/2-digit-Multiplication_<MODEL>_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl` |
| Mult probe | `results/open-r1/math_results/2-digit-Multiplication_<MODEL>_8888_corruption_FIX_force_b_rdelta9_n3000_probe_TRUTH_v2.probe.pt` |

Probe directories can be overridden with `TWOHOP_PROBE_DIR`, `HINT_PROBE_DIR` and `MULT_PROBE_DIR`.

**Data splits:**
- Every task is split train / validation / test = 60 / 20 / 20.
- TwoHop uses split seed 42; Hint and Mult use split seed 8888.
- Training uses only the train split, model selection uses only validation, and results are reported on test.
- Mult must use the `_probesplit` prompts built by `scripts/build_mult_probesplit_dataset.py`. They are aligned with the evaluation's probe split, so no evaluation prompt leaks into training. `train_offline.sh` builds them automatically.

## 6. Training

```bash
export SCRATCH=/path/to/your/scratch
TASK=twohop MODEL=Qwen3-8B              bash run/train_offline.sh
TASK=hint   MODEL=gemma-2-9b-it         bash run/train_offline.sh
TASK=mult   MODEL=Llama-3.1-8B-Instruct bash run/train_offline.sh
```

- Both RS-B and DPO-A are trained by default. Use `VARIANTS="rsB"` or `VARIANTS="dpoA"` to run only one.
- Sampling per task: TwoHop draws 1800 random train prompts with up to 1024 new tokens. Hint uses all 1800 train prompts with up to 768 tokens. Mult uses the whole train split with up to 512 tokens.
- Set `ROLLOUT_SEED` to draw an independent rollout sample, for seed replicates.
- Every step skips work whose output already exists, so an interrupted run can simply be restarted.
- Outputs go to `$SCRATCH/open-r1/cia/<run>`, where `<run>` is the recipe's `output_dir`, for example `two_hop_qwen3_8b_rs_B_v3`, `hint_gemma_9b_full_static_dpo_A_pilot` or `multiplication_llama31_8b_static_rs_B`.

## 7. Evaluation

Evaluate the base model first as the reference, then the trained run:

```bash
# TwoHop
MODEL=Qwen3-8B RUN=base                        bash run/eval_twohop.sh
MODEL=Qwen3-8B RUN=two_hop_qwen3_8b_rs_B_v3    bash run/eval_twohop.sh

# Hint
MODEL=gemma-2-9b-it RUN=base                                   bash run/eval_hint.sh
MODEL=gemma-2-9b-it RUN=hint_gemma_9b_full_static_dpo_A_pilot  bash run/eval_hint.sh

# Mult: generate first, then compute the primary causal metric
MODEL=Llama-3.1-8B-Instruct RUN=base TAG=base bash run/eval_mult_causal.sh
MODEL=Llama-3.1-8B-Instruct RUN=multiplication_llama31_8b_static_rs_B bash run/eval_mult.sh
MODEL=Llama-3.1-8B-Instruct RUN=multiplication_llama31_8b_static_rs_B TAG=final bash run/eval_mult_causal.sh
```

**Protocol, shared by all tasks:**
- Every checkpoint is evaluated on validation with generation seed 8888. **The peak is selected on validation only.**
- The peak and the last checkpoint are evaluated on test with three generation seeds: 8888, 5555 and 7777.
- Decoding: T = 0.7, top-p 0.95, seeded.
- Report ΔCIA = trained model − base model on the same seed. Significance comes from a paired sample-level bootstrap with 1000 resamples: `bash run/paired_bootstrap.sh <trained_labels.jsonl> <base_labels.jsonl>`.

**Evaluation sizes:**
- TwoHop samples 1000 validation and 2000 test rows.
- Hint uses the single-turn test rows that can be labelled, about 580 per model.
- Mult uses the test rows of the probe split that give an approach-B answer the corruption test can run on, about 290 to 590 per model (fewest for Llama).

**Outputs:** `$SCRATCH/results/open-r1/{twohop_eval_v3, hint_eval_v2, mult_eval_v2}/<run>/`. Each `<ckpt>_<split>_s<seed>.json` is a summary, and each `*labels.jsonl` holds per-row labels.

**Reporting Mult:** the causal metric is primary and the probe-based CIA is secondary; on trained models the two can disagree in sign. Report every Mult result as a full row: macro-F1, agreement, tracked rate, B_CoT rate, the four (B_INT, B_CoT) cells, and accuracy. Count a result as a real improvement only when the tracked rate rises, the (0,1) cell falls and accuracy does not collapse. macro-F1 alone is misleading in both directions: it penalizes a model that is almost always faithful, and it rewards a model that degenerates into the (0,0) cell.

