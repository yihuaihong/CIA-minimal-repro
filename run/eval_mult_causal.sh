#!/bin/bash
# Mult evaluation, step 2 (setup S-MULT-CAUSAL-EVAL-v1) — the PRIMARY Mult metric.
# Re-runs the partial-product corruption test on the TEST generations from run/eval_mult.sh, with the SAME
# weights that produced them:
#   B_INT := does the final answer follow the corrupted partial products (causal use of the written steps)
#   B_CoT := self-consistency (final answer == pp1 + pp2)
# Reports CIA_truth (macro-F1), tracked rate, the four (B_INT, B_CoT) cells and accuracy; when base labels of the
# same seed exist, also a paired bootstrap against the base model.
# Report every Mult result as a full row: macro-F1, agreement, tracked rate, B_CoT rate, cells, accuracy.
# A real gain needs tracked ↑, the (0,1) cell ↓ and accuracy not collapsed; macro-F1 alone can mislead.
#
# Usage:
#   MODEL=Qwen3-8B RUN=base TAG=base bash run/eval_mult_causal.sh                     # run this first (base reference)
#   MODEL=Qwen3-8B RUN=multiplication_qwen3_8b_static_rs_B TAG=final bash run/eval_mult_causal.sh
#   MODEL=Qwen3-8B RUN=<run> TAG=checkpoint-120 bash run/eval_mult_causal.sh
#
# Base-model generations expected by RUN=base:
#   seed 8888      : $SCRATCH/results/open-r1/math_results/2-digit-Multiplication_<MODEL>_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl
#   seeds 5555/7777: $SCRATCH/results/open-r1/math_results/mult_base_<MODEL>_gen_s<seed>.jsonl
#                    (produce with: python scripts/mult_gen_seed.py --model_name <MODEL> --gen_seed <seed>)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
: "${MODEL:?}" "${RUN:?}" "${TAG:?}"
SEEDS=${SEEDS:-"8888 5555 7777"}
MR=$SCRATCH/results/open-r1/math_results
OUTD=$SCRATCH/results/open-r1/mult_eval_v2/$RUN; [ "$RUN" = base ] && OUTD=${OUTD}_${MODEL}; mkdir -p "$OUTD"
if [ "$RUN" = base ]; then MDIR=$SCRATCH/transformers; MNAME=$MODEL
elif [ "$TAG" = final ]; then MDIR=$CKD; MNAME=$RUN            # final = the run directory itself
else MDIR=$CKD/$RUN; MNAME=$TAG; fi

for sd in $SEEDS; do
  causal=$OUTD/${TAG}_test_s${sd}_causal.jsonl; lab=$OUTD/${TAG}_test_s${sd}_truthlabels.jsonl
  if [ "$RUN" = base ]; then
    # restrict the 3000-row base files to the probe TEST rows; seed 8888 = the corruption file itself (already labelled)
    src=$MR/mult_base_${MODEL}_gen_s${sd}.jsonl; [ "$sd" = 8888 ] && src=$MR/2-digit-Multiplication_${MODEL}_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl
    dst=$OUTD/base_test_s${sd}_gen.jsonl; [ "$sd" = 8888 ] && dst=$causal
    [ -s "$dst" ] || python - "$MODEL" "$src" "$dst" <<'PY'
import json, sys
from scripts.build_mult_probesplit_dataset import probe_split_indices
M, src, dst = sys.argv[1:4]; tr, va, te = probe_split_indices(M); n = 0
with open(dst, "w") as f:
    for l in open(src):
        r = json.loads(l)
        if r["index"] in te: f.write(json.dumps(r) + "\n"); n += 1
print("base test rows", n, "->", dst)
PY
    gen=$dst; [ "$sd" = 8888 ] && gen=""
  else
    gen=$OUTD/${TAG}_test_s${sd}_gen.jsonl
  fi
  if [ -n "$gen" ] && [ ! -s "$causal" ]; then
    [ -s "$gen" ] || { echo "missing generation $gen — run run/eval_mult.sh first"; exit 1; }
    python -m CPF_utils.multiplication_corruption --input "$gen" --output "$causal" --model_dir "$MDIR" --model_name "$MNAME" \
        --prompt_variant force_b --input_format prompt_smoke --delta_range=-9,9 --delta_seed 8888 --resume
  fi
  python scripts/mult_causal_labels.py "$causal" "$lab"
  base=$SCRATCH/results/open-r1/mult_eval_v2/base_${MODEL}/base_test_s${sd}_truthlabels.jsonl
  if [ "$RUN" != base ] && [ -s "$base" ]; then
    python scripts/cia_bootstrap_v3.py "$lab" --vs "$base" > "$OUTD/${TAG}_test_s${sd}_causal.json"
    echo "$TAG s$sd CAUSAL paired vs base: $(cat "$OUTD/${TAG}_test_s${sd}_causal.json")"
  else
    python scripts/cia_bootstrap_v3.py "$lab" > "$OUTD/${TAG}_test_s${sd}_causal.json"
    echo "$TAG s$sd CAUSAL: $(cat "$OUTD/${TAG}_test_s${sd}_causal.json")"
  fi
done
echo "=== DONE mult causal eval $RUN $TAG ==="
