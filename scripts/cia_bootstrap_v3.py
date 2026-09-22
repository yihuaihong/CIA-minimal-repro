"""Paired sample-level bootstrap (N=1000, seed 8888) for CIA = macro-F1(B_INT, B_CoT) under the v3/v2 definitions.

Input row formats (auto-detected):
  TwoHop  twohop_eval_v3/<run>/<ckpt>_<split>.jsonl : {"i", "cell": [b_int, b_cot], "acc"}
  Hint    hint_*_v2labels.jsonl                      : {"index", "b_int_v2", "b_cot_v2", "has_cpos", "grpo_split", "hint_type", ...}
  Mult    mult_*_v2labels.jsonl                      : {"index", "b_int_v2", "b_cot_v2", "approach", "follows_partial_products", "acc_v2"}
Usage:
  python scripts/cia_bootstrap_v3.py FILE                       → CIA ± std, 95% CI
  python scripts/cia_bootstrap_v3.py FILE --vs BASE_FILE        → paired ΔCIA, CI, two-sided p (rows matched by key)
Filters: --hint_type suggestion_False (Hint), Mult always approach==B ∧ labelled, Hint always grpo_split==test ∧ has_cpos.
"""
import argparse, json, sys
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load(path, hint_type=None):
    rows = [json.loads(l) for l in open(path)]
    d = {}
    for r in rows:
        if "cell" in r:
            d[r["i"]] = (int(r["cell"][0]), int(r["cell"][1]), float(r.get("acc", 0)))
        elif "has_cpos" in r:
            if r.get("grpo_split") != "test" or not r.get("has_cpos"):
                continue
            if hint_type and r.get("hint_type") != hint_type:
                continue
            if r.get("b_cot_v2") is None:      # judge could not classify → excluded
                continue
            # key by prompt TEXT: `index` spaces differ between base files and restricted per-ckpt evals
            from CPF_utils.hint_bint import prompt_plain_text
            d[prompt_plain_text(r["biased_prompt"]).strip()] = (int(r["b_int_v2"]), int(r["b_cot_v2"]), float(r.get("acc_biased", 0)))
        else:
            if r.get("approach") != "B":      # probe-labelled population = approach-B rows (truth label optional)
                continue
            d[r["index"]] = (int(r["b_int_v2"]), int(r["b_cot_v2"]), float(r.get("acc_v2", 0)))
    return d


def macro_f1(b, c):
    f = []
    for pos in (0, 1):
        tp = np.sum((b == pos) & (c == pos)); fp = np.sum((b == pos) & (c != pos)); fn = np.sum((b != pos) & (c == pos))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("--vs", default=None); ap.add_argument("--hint_type", default=None)
    ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=8888)
    a = ap.parse_args()
    A = load(a.file, a.hint_type)
    rng = np.random.default_rng(a.seed)
    if a.vs:
        B = load(a.vs, a.hint_type)
        keys = sorted(set(A) & set(B))
        bA = np.array([A[k][0] for k in keys]); cA = np.array([A[k][1] for k in keys])
        bB = np.array([B[k][0] for k in keys]); cB = np.array([B[k][1] for k in keys])
        accA = np.mean([A[k][2] for k in keys]); accB = np.mean([B[k][2] for k in keys])
        d0 = macro_f1(bA, cA) - macro_f1(bB, cB); n = len(keys); ds = []
        for _ in range(a.n_boot):
            idx = rng.integers(0, n, n)
            ds.append(macro_f1(bA[idx], cA[idx]) - macro_f1(bB[idx], cB[idx]))
        ds = np.array(ds); p = 2 * min(np.mean(ds <= 0), np.mean(ds >= 0))
        print(json.dumps({"n": n, "CIA": round(macro_f1(bA, cA), 4), "CIA_base": round(macro_f1(bB, cB), 4),
                          "dCIA": round(d0, 4), "dCIA_std": round(float(ds.std()), 4),
                          "dCIA_CI95": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
                          "p": round(float(p), 4), "acc": round(float(accA), 4), "acc_base": round(float(accB), 4)}))
    else:
        keys = sorted(A); b = np.array([A[k][0] for k in keys]); c = np.array([A[k][1] for k in keys]); n = len(keys)
        vals = []
        for _ in range(a.n_boot):
            idx = rng.integers(0, n, n); vals.append(macro_f1(b[idx], c[idx]))
        vals = np.array(vals)
        print(json.dumps({"n": n, "CIA": round(macro_f1(b, c), 4), "std": round(float(vals.std()), 4),
                          "CI95": [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)],
                          "acc": round(float(np.mean([A[k][2] for k in keys])), 4)}))


if __name__ == "__main__":
    main()
