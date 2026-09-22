#!/bin/bash
# Mult evaluation, step 1 (setup S-MULT-EVAL-v2): generations + probe-based CIA for one training run.
#   every checkpoint-* (+ the final model) → VAL rows, gen seed 8888
#   VAL-peak + last checkpoint → TEST rows, gen seeds 8888 / 5555 / 7777
# Decoding: T = 0.7, top-p 0.95, 512 tokens, seeded.
# B_INT = truth probe v2 (pre-summation position), B_CoT = self-consistency (final answer == pp1 + pp2).
#
# The PRIMARY Mult metric is the causal one: run run/eval_mult_causal.sh afterwards on the TEST generations
# this script writes. The probe-based CIA printed here is secondary.
#
# Usage:
#   MODEL=Qwen3-8B RUN=multiplication_qwen3_8b_static_rs_B bash run/eval_mult.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
: "${MODEL:?}" "${RUN:?}"
SEEDS=${SEEDS:-"8888 5555 7777"}
DIR=$CKD/$RUN; OUTD=$SCRATCH/results/open-r1/mult_eval_v2/$RUN; mkdir -p "$OUTD"

ev() { # weights tag split seed
  [ -f "$1/config.json" ] || { echo "skip $1 (no weights)"; return; }
  local gen=$OUTD/$2_$3_s$4_gen.jsonl lab=$OUTD/$2_$3_s$4_v2labels.jsonl
  [ -s "$lab" ] && { echo "skip $lab"; return; }
  python scripts/mult_gen_seed.py --model_name "$MODEL" --ckpt "$1" --gen_seed "$4" --split "$3" --out "$gen"
  python scripts/mult_eval_label_v2.py --model_name "$MODEL" --ckpt "$1" --results "$gen" --out "$lab" --all_rows
  python scripts/cia_bootstrap_v3.py "$lab" > "$OUTD/$2_$3_s$4.json"; echo "$2 $3 s$4: $(cat "$OUTD/$2_$3_s$4.json")"
}
ev_test() { for sd in $SEEDS; do ev "$1" "$2" test "$sd"; done; }

CKPTS=$(for c in $(ls "$DIR" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n); do [ -f "$DIR/$c/config.json" ] && echo "$DIR/$c"; done || true)
for c in $CKPTS; do ev "$c" "$(basename "$c")" val 8888; done
[ -f "$DIR/config.json" ] && ev "$DIR" final val 8888

PEAK=$(python3 - "$OUTD" <<'EOF'
import json, sys, glob, os
best = None
for f in glob.glob(os.path.join(sys.argv[1], "*_val_s8888.json")):
    d = json.load(open(f)); tag = os.path.basename(f)[:-len("_val_s8888.json")]
    if best is None or d["CIA"] > best[1]: best = (tag, d["CIA"])
print(best[0] if best else "")
EOF
)
echo "peak VAL: $PEAK"
[ -n "$PEAK" ] && { W=$DIR/$PEAK; [ "$PEAK" = final ] && W=$DIR; ev_test "$W" "$PEAK"; }
LAST=$(echo "$CKPTS" | tail -1); LAST=${LAST:+$(basename "$LAST")}
[ -n "$LAST" ] && [ "$LAST" != "$PEAK" ] && ev_test "$DIR/$LAST" "$LAST"
echo "=== DONE ($RUN) — next: run/eval_mult_causal.sh with TAG=$PEAK ==="
