#!/bin/bash
# Shared environment for every script in run/. Source it; do not execute it.
#
# Required:
#   SCRATCH   root of the data/model layout described in README.md
# Optional:
#   NUM_GPUS  GPUs used for training (default 2)

: "${SCRATCH:?Set SCRATCH to the directory holding transformers/, open-r1/ and results/ (see README.md)}"

export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts:$REPO_ROOT/open-r1/src:${PYTHONPATH:-}"
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NUM_GPUS=${NUM_GPUS:-2}

# TwoHop B_INT: linear probe v3, K = 100 (setup S-TH-EVAL-v4)
export TWOHOP_PROBE_DIR=${TWOHOP_PROBE_DIR:-$SCRATCH/results/open-r1/probing_results/probe_chat_filtered_trainsplit_sp}
export TWOHOP_TOP_K=${TWOHOP_TOP_K:-100}
# Hint B_INT: probe v2 at the answer-letter position (setup S-HINT-EVAL-v2)
export HINT_PROBE_DIR=${HINT_PROBE_DIR:-$SCRATCH/results/open-r1/hint_mmlu_results}

DSD=$SCRATCH/open-r1/datasets        # HF datasets (task splits + built RS/DPO datasets)
ROLLD=$SCRATCH/open-r1/rollouts      # base-model rollouts and their scored versions
CKD=$SCRATCH/open-r1/cia             # training outputs (one directory per run)
mkdir -p "$DSD" "$ROLLD" "$CKD"

# Model name → short tag used in recipe and dataset names.
model_tag() {
  case "$1" in
    Qwen3-8B) echo qwen3_8b ;;
    gemma-2-9b-it) echo gemma_9b ;;
    Llama-3.1-8B-Instruct) echo llama31_8b ;;
    *) echo "unknown model $1 (expected Qwen3-8B | gemma-2-9b-it | Llama-3.1-8B-Instruct)" >&2; return 1 ;;
  esac
}
