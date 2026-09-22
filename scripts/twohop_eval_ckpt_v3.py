"""Per-checkpoint TwoHop evaluation under S-TH-EVAL-v3 (probe v3, K=100, entity-level
B_CoT, corrected cells) on the VAL or TEST split of the seed-42 shuffle.
  1. greedy generation (vLLM) of the CoT for each question (same PROMPT_TEMPLATE as eval)
  2. accuracy = FINAL ANSWER vs e3.value (+aliases), same rule as eval_twohop_greedy_2pos
  3. (B_INT, B_CoT) cells via CPF_utils.twohop_bint.corrected_cells on the SAME weights
  4. CIA = macro-F1; writes <out>.jsonl (per row) and <out>.json (summary)
Usage: python scripts/twohop_eval_ckpt_v3.py --model_name Qwen3-8B --weights <dir> --split val|test
        [--n_samples 1000] --out <path prefix>
"""
import argparse, ast, json, os, re, sys
import pandas as pd, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "open-r1", "src")]
_FINAL_RE = re.compile(r"FINAL ANSWER:?\s*(.+)", re.IGNORECASE)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def final_answer_correct(gen, gold, aliases):
    m = _FINAL_RE.search(gen or "")
    if not m:
        return 0
    pred = _norm(m.group(1).split("\n")[0])
    if not pred:
        return 0
    for g in [gold] + list(aliases):
        gn = _norm(g)
        if gn and (gn == pred or gn in pred or pred in gn):
            return 1
    return 0


def macro_f1(pairs):
    def f1(pos):
        tp = sum(1 for b, c in pairs if b == pos and c == pos)
        fp = sum(1 for b, c in pairs if b == pos and c != pos)
        fn = sum(1 for b, c in pairs if b != pos and c == pos)
        return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return (f1(1) + f1(0)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--n_samples", type=int, default=0, help="0 = full split; eval convention: test 2000")
    ap.add_argument("--out", required=True, help="output prefix (writes .jsonl and .json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    # decoding = paper appendix: nucleus sampling T=0.7, top-p 0.95, 512 tokens; gen_seed = eval seed dimension
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gen_seed", type=int, default=8888)
    a = ap.parse_args()
    S = os.environ["SCRATCH"]
    from eval_twohop_greedy_2pos import PROMPT_TEMPLATE
    from CPF_utils.chat import load_tokenizer, user_chat
    from CPF_utils.twohop_bint import TwoHopBIntProbe, default_probe_path, TOP_K_DEFAULT
    df = pd.read_csv(f"{S}/datasets/TwoHopFact/TwoHopFact.csv").sample(frac=1, random_state=a.seed).reset_index(drop=True)
    n_tr, n_va = int(len(df) * .6), int(len(df) * .2)
    part = df.iloc[n_tr:n_tr + n_va] if a.split == "val" else df.iloc[n_tr + n_va:]
    part = part.reset_index(drop=True)
    if 0 < a.n_samples < len(part):
        part = part.iloc[:a.n_samples].reset_index(drop=True)
    qs = part["r2(r1(e1)).prompt"].astype(str).tolist()
    e1s, e2s, e3s = (part[c].astype(str).tolist() for c in ("e1.value", "e2.value", "e3.value"))
    def aliases(s, fb):
        try:
            t = ast.literal_eval(s); out = []
            for grp in t:
                out.extend(str(x) for x in grp)
            return out or [fb]
        except Exception:
            return [fb]
    e3al = [aliases(s, fb) for s, fb in zip(part.get("e3.aliases", pd.Series([""] * len(part))).astype(str), e3s)]
    prompts = [PROMPT_TEMPLATE.format(question=q) for q in qs]

    tok = load_tokenizer(a.model_name)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.weights, tokenizer=f"{S}/transformers/{a.model_name}", dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
    outs = llm.generate([user_chat(tok, p) for p in prompts], SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.gen_seed))
    gens = [o.outputs[0].text for o in outs]
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.weights, dtype=torch.bfloat16).cuda().eval()
    probe = TwoHopBIntProbe(default_probe_path(a.model_name))
    cells = []
    for s in range(0, len(prompts), a.batch_size):
        cells += probe.corrected_cells(model, tok, prompts[s:s + a.batch_size], e1s[s:s + a.batch_size],
                                       e2s[s:s + a.batch_size], gens[s:s + a.batch_size], k=TOP_K_DEFAULT)
    accs = [final_answer_correct(g, e3, al) for g, e3, al in zip(gens, e3s, e3al)]
    from collections import Counter
    dist = Counter(cells); n = len(cells)
    summary = {"model_name": a.model_name, "weights": a.weights, "split": a.split, "n": n, "K": TOP_K_DEFAULT,
               "probe": probe.path, "setup_id": "S-TH-EVAL-v4", "decoding": {"temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens, "gen_seed": a.gen_seed}, "CIA": macro_f1(cells), "acc": sum(accs) / n,
               "cells": {str(k): v / n for k, v in sorted(dist.items())}}
    with open(a.out + ".jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "e1": e1s[i], "gold_bridge": e2s[i], "gen": gens[i], "cell": list(cells[i]),
                                "acc": accs[i]}) + "\n")
    json.dump(summary, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
