#!/bin/bash
# Offline training: RS-B and DPO-A for one task × one model.
#
#   1. base-model rollouts on the train split: 16 per prompt, T = 1.0, top-p 1.0
#   2. label every rollout with the SAME definition the evaluation uses
#      (B_INT from the probe on the base model, B_CoT from the task's verbalization check)
#   3. build the datasets
#        RS-B  : keep every rollout with faith = 1 (B_INT == B_CoT), accuracy not required  → SFT
#        DPO-A : per prompt, top-3 vs bottom-3 rollouts ranked by acc + faith             → DPO
#   4. train with accelerate + DeepSpeed ZeRO-3
#
# Usage:
#   TASK=twohop MODEL=Qwen3-8B bash run/train_offline.sh
#   TASK=hint   MODEL=gemma-2-9b-it VARIANTS="dpoA" bash run/train_offline.sh
#   TASK=mult   MODEL=Llama-3.1-8B-Instruct bash run/train_offline.sh
#
# Variables:
#   TASK       twohop | hint | mult
#   MODEL      Qwen3-8B | gemma-2-9b-it | Llama-3.1-8B-Instruct   (weights at $SCRATCH/transformers/$MODEL)
#   VARIANTS   default "rsB dpoA"
#   ROLLOUT_SEED  optional vLLM seed for the rollouts (used for rollout-seed replicates)
#   N_PROMPTS  hint only: train prompts to sample (default 1800 = full train split)
#   JUDGE_GPU  hint only: GPU for the Qwen2.5-32B judge server (default: the last visible GPU index = NUM_GPUS)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"

: "${TASK:?TASK=twohop|hint|mult}" "${MODEL:?MODEL=Qwen3-8B|gemma-2-9b-it|Llama-3.1-8B-Instruct}"
VARIANTS=${VARIANTS:-"rsB dpoA"}
TAG=$(model_tag "$MODEL")
MPATH=$SCRATCH/transformers/$MODEL
SEEDARG=${ROLLOUT_SEED:+--seed $ROLLOUT_SEED}

case $TASK in
  twohop)
    SRC=$DSD/TwoHopFact_cia_${MODEL}_linear_probe_v3_with_inner_subj
    ROLL=$ROLLD/two_hop_${TAG}_v3.jsonl; SCORED=${ROLL%.jsonl}_v3scored.jsonl; FO=${SCORED%.jsonl}_faithonly.jsonl
    RSB_DS=$DSD/TwoHopFact_rs_B_v3_${TAG};  DPOA_DS=$DSD/TwoHopFact_dpo_v3_${TAG}
    RSB_REC=recipes/CIA/rs/two_hop_${TAG}_v3_B.yaml; DPOA_REC=recipes/CIA/dpo/two_hop_${TAG}_v3.yaml
    [ -s "$ROLL" ] || (cd open-r1 && CUDA_VISIBLE_DEVICES=0 python scripts/cia/generate_rollouts.py --model_path "$MPATH" \
        --dataset_path "$SRC" --task two_hop --output "$ROLL" --num_rollouts 16 --max_new_tokens 1024 \
        --temperature 1.0 --top_p 1.0 --subsample 1800 $SEEDARG --wandb_project "")
    [ -s "$SCORED" ] || CUDA_VISIBLE_DEVICES=0 python scripts/score_twohop_rollouts_v3.py --model_name "$MODEL" \
        --rollouts "$ROLL" --dataset "$SRC" --out "$SCORED"
    [ -s "$FO" ] || python scripts/make_faithonly_rollouts.py "$SCORED" "$FO"
    ;;
  hint)
    SRC=$DSD/Hint_MMLU_cia; N_PROMPTS=${N_PROMPTS:-1800}; HTAG=${TAG}_full
    ROLL=$ROLLD/hint_${HTAG}_static_pilot.jsonl; SCORED=${ROLL%.jsonl}_v2scored.jsonl; FO=${SCORED%.jsonl}_faithonly.jsonl
    RSB_DS=$DSD/Hint_MMLU_static_rs_B_${HTAG}; DPOA_DS=$DSD/Hint_MMLU_static_dpo_A_${HTAG}
    RSB_REC=recipes/CIA/rs/hint_${HTAG}_static_pilot_B.yaml; DPOA_REC=recipes/CIA/dpo/hint_${HTAG}_static_pilot_A.yaml
    [ -s "$ROLL" ] || (cd open-r1 && CUDA_VISIBLE_DEVICES=0 python scripts/cia/generate_rollouts.py --model_path "$MPATH" \
        --dataset_path "$SRC" --task hint --output "$ROLL" --num_rollouts 16 --max_new_tokens 768 \
        --temperature 1.0 --top_p 1.0 --subsample "$N_PROMPTS" $SEEDARG --wandb_project "")
    if [ ! -s "$SCORED" ]; then
      # B_CoT for Hint comes from an LLM judge (Qwen2.5-32B-Instruct) served by vLLM on its own GPU.
      source scripts/judge_server.sh "${JUDGE_GPU:-$NUM_GPUS}" 8765
      trap 'kill $JUDGE_PID 2>/dev/null || true' EXIT
      CUDA_VISIBLE_DEVICES=0 python scripts/score_hint_rollouts_v2.py --model_name "$MODEL" \
          --rollouts "$ROLL" --dataset "$SRC" --out "$SCORED"      # also writes $FO
      kill $JUDGE_PID 2>/dev/null || true
    fi
    ;;
  mult)
    SRC=$DSD/Mult2d_cia_${MODEL}_probesplit
    ROLL=$ROLLD/mult_${TAG}_static.jsonl; SCORED=${ROLL%.jsonl}_v2scored.jsonl; FO=${SCORED%.jsonl}_faithonly.jsonl
    RSB_DS=$DSD/Mult2d_static_rs_B_${TAG}; DPOA_DS=$DSD/Mult2d_static_dpo_A_${TAG}
    RSB_REC=recipes/CIA/rs/multiplication_${TAG}_static_B.yaml; DPOA_REC=recipes/CIA/dpo/multiplication_${TAG}_static_A.yaml
    # leak-free prompt split aligned with the evaluation's probe split
    [ -f "$SRC/dataset_dict.json" ] || python scripts/build_mult_probesplit_dataset.py --model_name "$MODEL"
    [ -s "$ROLL" ] || (cd open-r1 && CUDA_VISIBLE_DEVICES=0 python scripts/cia/generate_rollouts.py --model_path "$MPATH" \
        --dataset_path "$SRC" --task multiplication --output "$ROLL" --num_rollouts 16 --max_new_tokens 512 \
        --temperature 1.0 --top_p 1.0 $SEEDARG --wandb_project "")
    [ -s "$SCORED" ] || CUDA_VISIBLE_DEVICES=0 python scripts/score_mult_rollouts_v2.py --model_name "$MODEL" \
        --rollouts "$ROLL" --out "$SCORED"                             # also writes $FO
    ;;
  *) echo "TASK must be twohop, hint or mult" >&2; exit 1 ;;
esac

# ── datasets ───────────────────────────────────────────────────────────────────────────────────────
[ -f "$RSB_DS/dataset_dict.json" ]  || (cd open-r1 && python scripts/cia/build_rs_dataset.py  --rollouts "$FO"     --out_dir "$RSB_DS" --tokenizer_path "$MPATH")
[ -f "$DPOA_DS/dataset_dict.json" ] || (cd open-r1 && python scripts/cia/build_dpo_dataset.py --rollouts "$SCORED" --out_dir "$DPOA_DS" --top_k 3)

# ── training ───────────────────────────────────────────────────────────────────────────────────────
train() { # entry recipe
  local ENTRY=$1 REC=$2
  local OUT; OUT=$(grep -E '^output_dir:' "open-r1/$REC" | head -1 | sed 's/^output_dir:\s*//' | tr -d '"' | sed "s|\${SCRATCH}|${SCRATCH}|g")
  [ -f "$OUT/config.json" ] && { echo "skip $REC (already trained: $OUT)"; return; }
  # a checkpoint interrupted mid-save has no trainer_state.json and would break resuming
  for ck in "$OUT"/checkpoint-*; do [ -d "$ck" ] && [ ! -f "$ck/trainer_state.json" ] && rm -rf "$ck"; done
  local RES; RES=$(mktemp --suffix=.yaml)
  sed "s|\${SCRATCH}|${SCRATCH}|g" "open-r1/$REC" > "$RES"
  echo "=== train $ENTRY $REC → $OUT ==="
  (cd open-r1 && accelerate launch --config_file recipes/accelerate_configs/zero3_no_offload_tuned.yaml \
      --num_processes "$NUM_GPUS" "src/open_r1/$ENTRY" --config "$RES")
  rm -f "$RES"
}
for V in $VARIANTS; do
  case $V in
    rsB)  train sft.py "$RSB_REC" ;;
    dpoA) train dpo.py "$DPOA_REC" ;;
    *) echo "unknown variant $V (use rsB and/or dpoA)" >&2; exit 1 ;;
  esac
done
echo "=== DONE $TASK $MODEL ($VARIANTS) ==="
