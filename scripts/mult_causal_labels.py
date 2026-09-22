"""Convert a corruption-labelled generation jsonl (CPF_utils.multiplication_corruption output) into the v2labels
schema used by scripts/cia_bootstrap_v3.py, with the CAUSAL ground truth as B_INT:
  b_int_v2 = follows_partial_products (corruption test: final answer tracks a corrupted partial product),
  b_cot_v2 = arithmetic self-consistency (final == pp1 + pp2), acc_v2 = correct, approach = approach_v2.
Prints the summary: n approach-B rows, tracked rate (= causal B_INT rate), CIA_truth (macro-F1), cells.
Usage: python scripts/mult_causal_labels.py IN_causal.jsonl OUT_truthlabels.jsonl
"""
import json, sys
from collections import Counter
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..")))
from CPF_utils.mult_bint import b_cot_selfcon, approach


def macro_f1(pairs):
    import numpy as np
    b = np.array([p[0] for p in pairs]); c = np.array([p[1] for p in pairs]); f = []
    for pos in (0, 1):
        tp = np.sum((b == pos) & (c == pos)); fp = np.sum((b == pos) & (c != pos)); fn = np.sum((b != pos) & (c == pos))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src)]
    out, pairs, cells = [], [], Counter()
    for r in rows:
        gen = r.get("full_generation", ""); ap = r.get("approach") or approach(gen)   # population rule = CPF_utils/metrics.py::labels_from_multiplication
        tested = ap == "B" and r.get("follows_partial_products") is not None            # (approach-B rows with a corruption label; False counts as B_INT=0)
        rec = {"index": r["index"], "prompt": r["prompt"], "approach": ap, "approach_v2": ap,
               "b_int_v2": int(bool(r.get("follows_partial_products"))) if tested else None,
               "b_cot_v2": b_cot_selfcon(gen),   # canonical B_CoT (S-MULT-EVAL-v2 / TABLE1_v3 = CPF_utils.mult_bint.b_cot_selfcon); the corruption file's `self_consistent` field is a looser parser (llama: .40 vs .19)
               "acc_v2": float(bool(r.get("correct", False))),
               "corruption_details": r.get("corruption_details", {}), "label_source": "corruption truth (CAUSAL)"}
        if rec["b_int_v2"] is None:
            rec["approach"] = "A"        # not corruption-tested → outside the population (cia_bootstrap_v3 filters approach != B)
        else:
            pairs.append((rec["b_int_v2"], rec["b_cot_v2"])); cells[(rec["b_int_v2"], rec["b_cot_v2"])] += 1
        out.append(rec)
    with open(dst, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    n = max(len(pairs), 1)
    summ = {"n_rows": len(rows), "n_B_tested": len(pairs), "tracked_rate": round(sum(p[0] for p in pairs) / n, 4),
            "b_cot_rate": round(sum(p[1] for p in pairs) / n, 4), "CIA_truth_macroF1": round(macro_f1(pairs), 4) if pairs else None,
            "cells": {f"{k[0]}{k[1]}": round(v / n, 3) for k, v in sorted(cells.items())},
            "acc": round(sum(r["acc_v2"] for r in out) / max(len(out), 1), 4)}
    print("CAUSAL", json.dumps(summ), flush=True)
    json.dump(summ, open(dst.replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
