"""Mult base generations for an extra evaluation seed (S-EVAL-DECODE-v1): vLLM nucleus sampling
T=0.7 / top-p 0.95 / 512 tokens, seeded; same force-B full prompt and question set as the n3000
corruption file (so rows align by `index`). Output jsonl fields: index, prompt (short), truth,
full_generation, approach (regex), plus final/acc for convenience.
Usage: python scripts/mult_gen_seed.py --model_name Qwen3-8B --gen_seed 5555 [--ckpt DIR] [--out F]
"""
import argparse, json, os, re, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "open-r1", "src")]
_FINAL = re.compile(r"FINAL ANSWER:?\s*(\d+)", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--gen_seed", type=int, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--split", choices=["all", "val", "test"], default="all",
                    help="restrict to the probe VAL / TEST rows (labelled approach-B rows, seed-8888 split)")
    a = ap.parse_args()
    S = os.environ["SCRATCH"]; MR = f"{S}/results/open-r1/math_results"
    from CPF_utils.chat import load_tokenizer, user_chat
    from CPF_utils.mult_bint import approach
    from scripts.train_mult_probe_truth import get_instruction
    src = f"{MR}/2-digit-Multiplication_{a.model_name}_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl"
    rows = [json.loads(l) for l in open(src)]
    if a.split != "all":   # probe split (seed-8888 permutation over labelled approach-B rows, as in train_mult_probe_truth)
        import numpy as np
        lab = [i for i, r in enumerate(rows) if r.get("approach") == "B" and r.get("follows_partial_products") is not None]
        rng = np.random.default_rng(8888); perm = rng.permutation(len(lab))
        ntr, nv = int(len(perm) * .6), int(len(perm) * .2)
        sel = perm[ntr:ntr + nv] if a.split == "val" else perm[ntr + nv:]
        keep = {rows[lab[p]]["index"] for p in sel}
        rows = [r for r in rows if r["index"] in keep]
    instr = get_instruction()
    tok = load_tokenizer(a.model_name)
    from vllm import LLM, SamplingParams
    weights = a.ckpt or f"{S}/transformers/{a.model_name}"
    llm = LLM(model=weights, tokenizer=f"{S}/transformers/{a.model_name}", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    outs = llm.generate([user_chat(tok, instr + r["prompt"]) for r in rows],
                        SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.gen_seed))
    tag = os.path.basename(a.ckpt.rstrip("/")) if a.ckpt else "base"
    out = a.out or f"{MR}/mult_{tag}_{a.model_name}_gen_s{a.gen_seed}.jsonl"
    with open(out, "w") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0].text; m = _FINAL.search(g); truth = r.get("truth") or r.get("correct_answer")
            f.write(json.dumps({"index": r["index"], "prompt": r["prompt"], "truth": truth, "full_generation": g,
                                "approach": approach(g), "final": int(m.group(1)) if m else None,
                                "correct": bool(m and truth is not None and int(m.group(1)) == int(truth)),
                                "decoding": {"temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens, "gen_seed": a.gen_seed},
                                "weights": weights}) + "\n")
    print("saved", out, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
