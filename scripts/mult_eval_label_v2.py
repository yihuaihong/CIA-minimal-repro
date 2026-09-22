"""Mult eval labelling v2 (S-MULT-EVAL-v2): for a generation results jsonl (base or
trained ckpt; fields prompt/full_generation/correct_answer or truth/index), compute
  b_int_v2   = truth-trained probe v2 on the model that produced the generation
  b_cot_v2   = arithmetic self-consistency (final == pp1 + pp2)
  approach   = B if ≥2 partial-product lines else A
  acc        = final == truth
via the single implementation CPF_utils.mult_bint. Rows are restricted to the
paper's eval population (approach B, seed-8888 20% test indices from the probe ckpt).
Usage: python scripts/mult_eval_label_v2.py --model_name Qwen3-8B --results <gen.jsonl> [--ckpt DIR] [--out F]
"""
import argparse, json, os, re, sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "open-r1", "src")]
_FINAL = re.compile(r"FINAL ANSWER:?\s*(\d+)", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--all_rows", action="store_true", help="label every row (default: test population only)")
    a = ap.parse_args()
    S = os.environ["SCRATCH"]
    from transformers import AutoModelForCausalLM
    from CPF_utils.chat import load_tokenizer
    from CPF_utils.mult_bint import MultBIntProbe, b_cot_selfcon, approach
    from scripts.train_mult_probe_truth import get_instruction
    import open_r1.live_probe_mult as lp

    rows = [json.loads(l) for l in open(a.results)]
    probe = MultBIntProbe(lp.default_probe_path(a.model_name))
    test_idx = set(probe.meta["test_indices"])
    if not a.all_rows:
        rows = [r for r in rows if r.get("index") in test_idx]
    instr = get_instruction()
    tok = load_tokenizer(a.model_name)
    weights = a.ckpt or f"{S}/transformers/{a.model_name}"
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16).cuda().eval()
    print(f"probe {probe.path} L{probe.layer} test_agreement={probe.meta.get('test_agreement')}; weights {weights}; rows {len(rows)}", flush=True)
    b_int = []
    for s in range(0, len(rows), a.batch_size):
        b = rows[s:s + a.batch_size]
        prompts = [(r["prompt"] if "Now solve" in r["prompt"] else instr + r["prompt"]) for r in b]
        b_int += probe.b_int(model, tok, prompts, [r["full_generation"] for r in b])
    out = a.out or a.results.replace(".jsonl", "_v2labels.jsonl")
    nB = 0
    with open(out, "w") as f:
        for r, bi in zip(rows, b_int):
            g = r["full_generation"]; m = _FINAL.search(g); truth = r.get("truth") or r.get("correct_answer")
            r.update(b_int_v2=int(bi), b_cot_v2=b_cot_selfcon(g), approach_v2=approach(g),
                     acc_v2=int(m is not None and truth is not None and int(m.group(1)) == int(truth)),
                     probe_v2_path=probe.path)
            nB += r["approach_v2"] == "B"
            f.write(json.dumps(r) + "\n")
    print(f"saved {out}: rows={len(rows)} approachB={nB} B_INT rate={sum(b_int)/len(b_int):.3f}", flush=True)


if __name__ == "__main__":
    main()
