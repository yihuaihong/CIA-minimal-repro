#!/bin/bash
# TwoHop evaluation (setup S-TH-EVAL-v4) for one training run.
#   every checkpoint-* → VAL (n = 1000, gen seed 8888)
#   VAL-peak checkpoint + last checkpoint → TEST (n = 2000, gen seeds 8888 / 5555 / 7777)
#   RUN=base evaluates the untrained model (VAL + TEST)
# Decoding: T = 0.7, top-p 0.95, 512 tokens, seeded. B_INT = probe v3 (K = 100), B_CoT = bridge entity said in the CoT.
#
# Usage:
#   MODEL=Qwen3-8B RUN=two_hop_qwen3_8b_rs_B_v3 bash run/eval_twohop.sh
#   MODEL=Qwen3-8B RUN=base bash run/eval_twohop.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
: "${MODEL:?}" "${RUN:?}"
SEEDS=${SEEDS:-"8888 5555 7777"}
OUTD=$SCRATCH/results/open-r1/twohop_eval_v3/$RUN; [ "$RUN" = base ] && OUTD=${OUTD}_${MODEL}; mkdir -p "$OUTD"

ev() { # weights tag split n seed
  local out=$OUTD/$2_$3_s$5
  [ -s "$out.json" ] && { echo "skip $out"; return; }
  python scripts/twohop_eval_ckpt_v3.py --model_name "$MODEL" --weights "$1" --split "$3" --n_samples "$4" --out "$out" --gen_seed "$5"
}
ev_test() { for sd in $SEEDS; do ev "$1" "$2" test 2000 "$sd"; done; }

if [ "$RUN" = base ]; then
  W=$SCRATCH/transformers/$MODEL; ev "$W" base val 1000 8888; ev_test "$W" base; exit 0
fi

DIR=$CKD/$RUN
CKPTS=$(ls "$DIR" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | sed "s|^|$DIR/|" || true)
for c in $CKPTS; do ev "$c" "$(basename "$c")" val 1000 8888; done
[ -f "$DIR/config.json" ] && ev "$DIR" final val 1000 8888

# model selection on VAL only
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

python3 - "$OUTD" <<'EOF'
import json, sys, glob, os
print("| ckpt_split | split | CIA | acc | n | seed |"); print("|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    print("| " + " | ".join(map(str, (os.path.basename(f)[:-5], d["split"], round(d["CIA"], 3), round(d["acc"], 3), d["n"], d.get("decoding", {}).get("gen_seed")))) + " |")
EOF
echo "=== DONE ($RUN) ==="
