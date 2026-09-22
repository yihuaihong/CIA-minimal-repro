"""Train the Mult B_INT linear probe on corruption GROUND-TRUTH labels (S-MULT-PROBE-v2).

Rows: corruption jsonl (n3000), approach B with follows_partial_products not None.
Split: the SAME seed-8888 60/20/20 permutation make_final_table.mult_testidx uses
(so the 20% test rows = the paper's Mult eval population; never trained on).
Features: last-token hidden state of chat(full prompt) + pre-summation CoT prefix
(CPF_utils.mult_bint.probe_text) at every layer of a sweep; standardize with
train mean/std; logistic probe (nn.Linear(H,2), CE, AdamW); select layer+HP on val
macro-F1; report test macro-F1 / agreement with the truth label.
Output ckpt schema == the old probes (layer, mean, std, state_dict, ...) so
MultBIntProbe loads it; plus `trained_with="TRUTH_full_prompt_v2"`.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from CPF_utils.chat import load_tokenizer  # noqa: E402
from CPF_utils.mult_bint import probe_text, MultBIntProbe  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

S = os.environ["SCRATCH"]
MR = f"{S}/results/open-r1/math_results"


def macro_f1(y, p):
    y, p = np.asarray(y), np.asarray(p)
    f = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def get_instruction():
    from datasets import load_from_disk
    import glob
    for d in glob.glob(f"{S}/open-r1/datasets/Mult2d_cia_*_force_b_rdelta9_strict"):
        sample = load_from_disk(d)["train"][0]["problem"]
        if "Now solve" in sample:
            return sample.rsplit("Now solve", 1)[0] + "Now solve the following multiplication:\n"
    raise RuntimeError("no force_b dataset found to derive the instruction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    f = f"{MR}/2-digit-Multiplication_{a.model_name}_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl"
    recs = [json.loads(l) for l in open(f)]
    lab = [i for i, r in enumerate(recs) if r.get("approach") == "B" and r.get("follows_partial_products") is not None]
    rng = np.random.default_rng(8888); perm = rng.permutation(len(lab))
    ntr, nv = int(len(perm) * .6), int(len(perm) * .2)
    split = {"train": perm[:ntr], "val": perm[ntr:ntr + nv], "test": perm[ntr + nv:]}
    instr = get_instruction()
    rows = [recs[i] for i in lab]
    texts, y = [], []
    tok = load_tokenizer(a.model_name)
    for r in rows:
        texts.append(probe_text(tok, instr + r["prompt"], r["full_generation"]))
        y.append(int(bool(r["follows_partial_products"])))
    y = np.array(y)
    print(f"{a.model_name}: labeled B rows={len(rows)} pos-rate={y.mean():.3f} split={ {k: len(v) for k, v in split.items()} }", flush=True)

    model = AutoModelForCausalLM.from_pretrained(f"{S}/transformers/{a.model_name}", dtype=torch.bfloat16).cuda().eval()
    nL = model.config.num_hidden_layers
    layers = a.layers or list(range(nL // 3, nL - 2, 2))
    dummy = MultBIntProbe.__new__(MultBIntProbe)   # reuse last_hidden without a ckpt
    feats = {L: [] for L in layers}
    for s in tqdm(range(0, len(texts), a.batch_size), desc="hidden"):
        h = MultBIntProbe.last_hidden(dummy, model, tok, texts[s:s + a.batch_size], layers=layers)
        for L in layers:
            feats[L].append(h[L])
    feats = {L: torch.cat(v) for L, v in feats.items()}
    del model; torch.cuda.empty_cache()

    best = None
    grid = [(lr, wd) for lr in (1e-3, 3e-4) for wd in (1e-2, 1e-1, 1.0)]
    for L in layers:
        X = feats[L]; tr, va = split["train"], split["val"]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Z = (X - mu) / sd
        for lr, wd in grid:
            torch.manual_seed(0)
            probe = nn.Linear(X.shape[1], 2); opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
            Ztr, ytr = Z[tr], torch.tensor(y[tr])
            for ep in range(200):
                opt.zero_grad(); loss = nn.functional.cross_entropy(probe(Ztr), ytr); loss.backward(); opt.step()
            with torch.no_grad():
                pv = probe(Z[va]).argmax(-1).numpy()
            f1 = macro_f1(y[va], pv)
            if best is None or f1 > best["val_f1"]:
                best = dict(layer=L, lr=lr, wd=wd, val_f1=f1, mu=mu, sd=sd, state=probe.state_dict())
        print(f"  layer {L}: best val F1 so far {best['val_f1']:.3f} (L{best['layer']})", flush=True)

    L = best["layer"]; Z = (feats[L] - best["mu"]) / best["sd"]
    probe = nn.Linear(feats[L].shape[1], 2); probe.load_state_dict(best["state"])
    with torch.no_grad():
        pt = probe(Z[split["test"]]).argmax(-1).numpy()
    test_f1 = macro_f1(y[split["test"]], pt); test_agree = float((pt == y[split["test"]]).mean())
    print(f"SELECTED layer={L} lr={best['lr']} wd={best['wd']} val_F1={best['val_f1']:.3f} "
          f"TEST macroF1={test_f1:.3f} agreement={test_agree:.3f} (truth pos-rate test={y[split['test']].mean():.3f})", flush=True)
    out = a.out or f"{MR}/2-digit-Multiplication_{a.model_name}_8888_corruption_FIX_force_b_rdelta9_n3000_probe_TRUTH_v2.probe.pt"
    torch.save({"layer": L, "mean": best["mu"], "std": best["sd"], "state_dict": probe.state_dict(),
                "model_name": a.model_name, "label_mode": "follows_partial_products (corruption truth)",
                "probe_position": "pre_summation", "trained_with": "TRUTH_full_prompt_v2",
                "split_seed": 8888, "val_f1": best["val_f1"], "test_f1": test_f1, "test_agreement": test_agree,
                "hp": {"lr": best["lr"], "wd": best["wd"], "epochs": 200}, "layers_swept": layers,
                "test_indices": [recs[lab[p]]["index"] for p in split["test"]]}, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
