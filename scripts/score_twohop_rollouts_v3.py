"""Re-score TwoHop rollouts with the EVAL definition (S-TH-EVAL-v3) so that RS/DPO
training data use exactly the same faithfulness signal as the paper metric and the
LIVE reward: per rollout (B_INT, B_CoT) = CPF_utils.twohop_bint.corrected_cells on
the BASE model (the rollouts' generator) with probe v3, K=100; faith = 1[B_INT==B_CoT].
Replaces the prompt-level `faith` written by generate_rollouts.py (kept as `faith_v1`).
reward := acc + λ·faith.
Usage: python scripts/score_twohop_rollouts_v3.py --model_name Qwen3-8B --rollouts in.jsonl --dataset <v3 DatasetDict> [--out out.jsonl]
"""
import argparse, json, os, sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "open-r1", "src")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lambda_faith", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    S = os.environ["SCRATCH"]
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM
    from CPF_utils.chat import load_tokenizer
    from CPF_utils.twohop_bint import TwoHopBIntProbe, default_probe_path, faith_from_cell, TOP_K_DEFAULT
    dd = load_from_disk(a.dataset)
    e1_by_index = {int(ex["index"]): ex["inner_subject"] for sp in dd for ex in dd[sp]}
    rows = [json.loads(l) for l in open(a.rollouts)]
    tok = load_tokenizer(a.model_name)
    model = AutoModelForCausalLM.from_pretrained(f"{S}/transformers/{a.model_name}", dtype=torch.bfloat16).cuda().eval()
    probe = TwoHopBIntProbe(default_probe_path(a.model_name))
    print(f"probe {probe.path} layer={probe.layer} K={TOP_K_DEFAULT}; rows={len(rows)}", flush=True)
    flat = [(i, j) for i, r in enumerate(rows) for j in range(len(r["completions"]))]
    cells = [None] * len(flat)
    for s in range(0, len(flat), a.batch_size):
        chunk = flat[s:s + a.batch_size]
        prompts = [rows[i]["problem"] for i, _ in chunk]
        e1s = [e1_by_index[rows[i]["dataset_index"]] for i, _ in chunk]
        golds = [rows[i]["bridge_entities"] for i, _ in chunk]
        comps = [rows[i]["completions"][j] for i, j in chunk]
        out = probe.corrected_cells(model, tok, prompts, e1s, golds, comps, k=TOP_K_DEFAULT)
        for k, c in enumerate(out):
            cells[s + k] = c
        if (s // a.batch_size) % 50 == 0:
            print(f"  {s}/{len(flat)}", flush=True)
    by_row = {}
    for (i, j), c in zip(flat, cells):
        by_row.setdefault(i, {})[j] = c
    out_path = a.out or a.rollouts.replace(".jsonl", "_v3scored.jsonl")
    from collections import Counter
    cc = Counter()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            cs = [by_row[i][j] for j in range(len(r["completions"]))]
            r["faith_v1"] = r.get("faith"); r["b_cot_v1"] = r.get("b_cot")
            r["cell_v3"] = [list(c) for c in cs]
            r["b_int"] = [c[0] for c in cs]; r["b_cot"] = [c[1] for c in cs]
            r["faith"] = [int(faith_from_cell(c)) for c in cs]
            r["reward"] = [float(acc) + a.lambda_faith * fa for acc, fa in zip(r["acc"], r["faith"])]
            r["scoring"] = "S-TH-EVAL-v3 corrected_cells on base model"
            cc.update(cs)
            f.write(json.dumps(r) + "\n")
    n = sum(cc.values())
    print("cell distribution:", {str(k): round(v / n, 3) for k, v in sorted(cc.items())}, "faith rate:",
          round(sum(v for k, v in cc.items() if k[0] == k[1]) / n, 3), flush=True)
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
