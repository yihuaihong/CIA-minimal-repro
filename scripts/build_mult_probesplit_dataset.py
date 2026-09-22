"""Build Mult2d_cia_<M>_probesplit: the force-B prompt DatasetDict re-split along the S-MULT-EVAL-v2 probe
split (seed-8888 permutation of the labelled approach-B rows of the base n3000 corruption file, 60/20/20), so
that offline-training rollouts (train) never touch the eval VAL/TEST rows.  The seed-42 Mult2d_cia_*_strict
split overlaps the probe TEST by ~59% (2026-09-13 audit) and must not be used for training.
Usage: python scripts/build_mult_probesplit_dataset.py --model_name gemma-2-9b-it
"""
import argparse, json, os
import numpy as np
from datasets import load_from_disk, concatenate_datasets, DatasetDict


def probe_split_indices(model_name):
    S = os.environ["SCRATCH"]
    src = f"{S}/results/open-r1/math_results/2-digit-Multiplication_{model_name}_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl"
    rows = [json.loads(l) for l in open(src)]
    lab = [i for i, r in enumerate(rows) if r.get("approach") == "B" and r.get("follows_partial_products") is not None]
    rng = np.random.default_rng(8888); perm = rng.permutation(len(lab))
    ntr, nv = int(len(perm) * .6), int(len(perm) * .2)
    idx = lambda sel: {rows[lab[p]]["index"] for p in sel}
    return idx(perm[:ntr]), idx(perm[ntr:ntr + nv]), idx(perm[ntr + nv:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model_name", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args(); S = os.environ["SCRATCH"]
    dd = load_from_disk(f"{S}/open-r1/datasets/Mult2d_cia_{a.model_name}_force_b_rdelta9_strict")
    allrows = concatenate_datasets([dd[sp] for sp in ("train", "validation", "test")])
    tr, va, te = probe_split_indices(a.model_name)
    assert not (tr & va) and not (tr & te) and not (va & te)
    new = DatasetDict({"train": allrows.filter(lambda ex: int(ex["index"]) in tr),
                       "validation": allrows.filter(lambda ex: int(ex["index"]) in va),
                       "test": allrows.filter(lambda ex: int(ex["index"]) in te)})
    out = a.out or f"{S}/open-r1/datasets/Mult2d_cia_{a.model_name}_probesplit"
    new.save_to_disk(out)
    print("saved", out, {sp: len(new[sp]) for sp in new}, flush=True)


if __name__ == "__main__":
    main()
