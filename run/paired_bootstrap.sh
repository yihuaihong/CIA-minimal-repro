#!/bin/bash
# Paired sample-level bootstrap (1000 resamples) of ΔCIA between a trained model and the base model on the
# same eval seed and the same rows. Works on any *_v2labels.jsonl / *_truthlabels.jsonl produced by run/eval_*.sh.
#
# Usage:
#   bash run/paired_bootstrap.sh <trained_labels.jsonl> <base_labels.jsonl>
# Prints JSON: n, CIA, CIA_base, dCIA, dCIA_std, dCIA_CI95, p, acc, acc_base.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
python scripts/cia_bootstrap_v3.py "${1:?trained labels}" --vs "${2:?base labels}"
