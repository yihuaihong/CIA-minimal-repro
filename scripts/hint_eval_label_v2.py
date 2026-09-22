"""Hint eval labelling v2 (S-HINT-EVAL-v2): per row B_INT (C-pos probe v2 on the
model that produced the generation) and B_CoT (v3 judge, offline vLLM) — both via
the single implementations in CPF_utils. TWO PROCESSES (a torn-down policy model
leaves ~14 GB of CUDA context, which starves the 32B judge of KV memory):
  --stage probe : policy/ckpt forward → <out>.probe.jsonl  (b_int_v2, has_cpos, splits, accs)
  --stage judge : Qwen2.5-32B judge   → <out>               (adds b_cot_v2)
  --stage all   : runs both as subprocesses (default)
Usage:
  python scripts/hint_eval_label_v2.py --model_name Qwen3-8B --results <results.jsonl> \
      [--ckpt <trained weights dir>] [--out <labeled.jsonl>] [--grpo_dataset DIR] [--judge_tp 1]
"""
import argparse, json, os, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "open-r1", "src")]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ckpt", default=None, help="trained weights (default: base model)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge_tp", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grpo_dataset", default=None,
                    help="HF DatasetDict used for GRPO; rows get grpo_split ∈ train/validation/test/not_in_dataset")
    ap.add_argument("--stage", choices=["probe", "judge", "all"], default="all")
    a = ap.parse_args()
    a.out = a.out or a.results.replace(".jsonl", "_v2labels.jsonl")
    a.probe_out = a.out + ".probe.jsonl"
    return a


def stage_probe(a):
    import torch
    from transformers import AutoModelForCausalLM
    from CPF_utils.chat import load_tokenizer
    from CPF_utils.hint_bint import HintBIntProbe, prompt_plain_text
    import open_r1.live_probe_hint as lp
    S = os.environ["SCRATCH"]
    rows = [json.loads(l) for l in open(a.results)]
    tok = load_tokenizer(a.model_name)
    weights = a.ckpt or f"{S}/transformers/{a.model_name}"
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16).cuda().eval()
    probe = HintBIntProbe(lp.hint_probe_path(a.model_name))
    print(f"probe {probe.path} layer={probe.layer}; weights {weights}; rows {len(rows)}", flush=True)
    b_int = []
    for s in range(0, len(rows), a.batch_size):
        b = rows[s:s + a.batch_size]
        b_int += probe.b_int(model, tok, [r["biased_prompt"] for r in b], [r.get("biased_generation", "") for r in b])
    n_cpos = sum(x is not None for x in b_int)
    print(f"stage probe done: with C-position {n_cpos}/{len(rows)}, B_INT rate {sum(x or 0 for x in b_int)/max(n_cpos,1):.3f}", flush=True)
    split_by_index, split_by_text = {}, {}
    if a.grpo_dataset:
        from datasets import load_from_disk
        dd = load_from_disk(a.grpo_dataset)
        for sp in dd:
            for ex in dd[sp]:
                split_by_index[str(ex["index"])] = sp
                split_by_text[ex["problem"].strip()] = sp
    with open(a.probe_out, "w") as f:
        for r, bi in zip(rows, b_int):
            # TEXT join only: results-jsonl `index` is per-format (posthoc i ↔ suggestion i+3000) and does
            # not match the GRPO dataset's index space; posthoc (3-turn) rows are never in the dataset.
            sp = split_by_text.get(prompt_plain_text(r["biased_prompt"]).strip())
            ca = str(r.get("correct_answer", "")).strip().upper(); h = str(r.get("hint", "")).strip().upper()
            r.update(b_int_v2=bi, has_cpos=bi is not None, probe_v2_path=probe.path,
                     grpo_split=sp or "not_in_dataset", hint_type=r.get("hint_type"),
                     acc_biased=int(str(r.get("pred_biased", "")).strip().upper() == ca),
                     acc_unbiased=int(str(r.get("pred_unbiased", "")).strip().upper() == ca),
                     followed_hint=int(str(r.get("pred_biased", "")).strip().upper() == h))
            f.write(json.dumps(r) + "\n")
    if a.grpo_dataset:
        from collections import Counter
        print("grpo_split counts:", Counter(json.loads(l)["grpo_split"] for l in open(a.probe_out)), flush=True)
    print("saved", a.probe_out, flush=True)


def stage_judge(a):
    from CPF_utils.hint_bint import prompt_plain_text
    from CPF_utils.hint_judge import judge_offline
    rows = [json.loads(l) for l in open(a.probe_out)]
    b_cot = judge_offline([prompt_plain_text(r["biased_prompt"]) for r in rows],
                          [r.get("hint", "") for r in rows],
                          [r.get("biased_generation", "") for r in rows], tp=a.judge_tp)
    ok = [b for b in b_cot if b is not None]
    print(f"stage judge done: B_CoT rate {sum(ok)/max(len(ok),1):.3f}; unclassified {len(b_cot)-len(ok)}/{len(b_cot)}", flush=True)
    with open(a.out, "w") as f:
        for r, bc in zip(rows, b_cot):
            r["b_cot_v2"] = None if bc is None else int(bc)   # None rows are excluded downstream
            f.write(json.dumps(r) + "\n")
    print("saved", a.out, flush=True)


def main():
    a = parse()
    if a.stage == "probe":
        stage_probe(a)
    elif a.stage == "judge":
        stage_judge(a)
    else:
        base = [sys.executable, os.path.abspath(__file__)] + [x for x in sys.argv[1:] if x not in ("--stage", "all")]
        base = [x for x in base if x != "--stage"]
        for st in ("probe", "judge"):
            print(f"=== running stage {st} in a fresh process ===", flush=True)
            subprocess.run(base + ["--stage", st], check=True)


if __name__ == "__main__":
    main()
