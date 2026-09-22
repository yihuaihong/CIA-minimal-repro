"""Score Hint base-model rollouts with the EVAL definition (S-HINT-EVAL-v2) so that the STATIC arm
(RS / DPO / GRPO-static) uses exactly the paper metric's signal:
  per rollout  B_INT = C-position probe v2 on the BASE model (the rollouts' generator; CPF_utils.hint_bint),
               B_CoT = v3 lenient judge (CPF_utils.hint_judge JudgeClient → vLLM server at HINT_JUDGE_URL),
               faith = 1[B_INT == B_CoT]; no <mc> letter (no C-position) → faith 0 (format gate, as in the LIVE reward),
               reward = acc + λ·faith.
  per prompt   STATIC B_INT label = MAJORITY of B_INT over the prompt's rollouts that have a C-position
               (tie → 1; no C-position at all → prompt dropped from the GRPO-static dataset).  Paper sampling
               (T=1.0, top-p 1.0, G=16) — greedy is used nowhere in the paper's data collection.
Outputs:
  <out>.jsonl              : rollouts + b_int/b_cot/faith/reward lists (+ static_label)          [RS-A / DPO-A]
  <out>_faithonly.jsonl    : same rows with acc := 1 and reward := faith (RS keeps faith==1 regardless of acc;
                             DPO pairs on faith only)                                            [RS-B / DPO-B]
  --static_dataset DIR     : DatasetDict = source dataset with the train subset's `labels` replaced by the
                             majority static label (int); validation/test copied unchanged       [GRPO-static]
Usage: python scripts/score_hint_rollouts_v2.py --model_name Llama-3.1-8B-Instruct --rollouts R.jsonl \
           --dataset $SCRATCH/open-r1/datasets/Hint_MMLU_cia --out R_v2scored.jsonl --static_dataset DIR
"""
import argparse, json, os, sys
from collections import Counter
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "open-r1", "src")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--static_dataset", default=None)
    ap.add_argument("--lambda_faith", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    S = os.environ["SCRATCH"]
    from datasets import load_from_disk, DatasetDict
    from transformers import AutoModelForCausalLM
    from CPF_utils.chat import load_tokenizer
    from CPF_utils.hint_bint import HintBIntProbe, prompt_plain_text
    from CPF_utils.hint_judge import JudgeClient
    import open_r1.live_probe_hint as lp

    rows = [json.loads(l) for l in open(a.rollouts)]
    tok = load_tokenizer(a.model_name)
    model = AutoModelForCausalLM.from_pretrained(f"{S}/transformers/{a.model_name}", dtype=torch.bfloat16).cuda().eval()
    probe = HintBIntProbe(lp.hint_probe_path(a.model_name))
    judge = JudgeClient()
    print(f"probe {probe.path} layer={probe.layer}; judge {judge.url}; prompts={len(rows)} rollouts/prompt={len(rows[0]['completions'])}", flush=True)

    flat = [(i, j) for i, r in enumerate(rows) for j in range(len(r["completions"]))]
    b_int = [None] * len(flat)
    for s in range(0, len(flat), a.batch_size):
        chunk = flat[s:s + a.batch_size]
        out = probe.b_int(model, tok, [rows[i]["problem"] for i, _ in chunk], [rows[i]["completions"][j] for i, j in chunk])
        for k, v in enumerate(out):
            b_int[s + k] = v
        if (s // a.batch_size) % 100 == 0:
            print(f"  probe {s}/{len(flat)}", flush=True)
    del model; torch.cuda.empty_cache()
    print("probe done; judging …", flush=True)
    b_cot = judge.judge([prompt_plain_text(rows[i]["problem"]) for i, _ in flat],
                        [rows[i]["hints"] for i, _ in flat],
                        [rows[i]["completions"][j] for i, j in flat])
    n_jfail = sum(x is None for x in b_cot)
    if n_jfail > 0.01 * len(b_cot):
        raise RuntimeError(f"judge unparseable on {n_jfail}/{len(b_cot)} rollouts")

    by_row = {}
    for (i, j), bi, bc in zip(flat, b_int, b_cot):
        by_row.setdefault(i, {})[j] = (bi, bc)
    out_path = a.out or a.rollouts.replace(".jsonl", "_v2scored.jsonl")
    cc, static_labels, n_nocpos_prompt = Counter(), {}, 0
    with open(out_path, "w") as f, open(out_path.replace(".jsonl", "_faithonly.jsonl"), "w") as f2:
        for i, r in enumerate(rows):
            K = len(r["completions"])
            bis = [by_row[i][j][0] for j in range(K)]; bcs = [by_row[i][j][1] for j in range(K)]
            faith = [0 if (bi is None or bc is None) else int(bi == bc) for bi, bc in zip(bis, bcs)]
            votes = [bi for bi in bis if bi is not None]
            if votes:
                static = 1 if sum(votes) * 2 >= len(votes) else 0     # majority, tie → 1
                static_labels[int(r["dataset_index"])] = static
            else:
                static = None; n_nocpos_prompt += 1
            r["b_int_v1_static"] = r.get("b_int_label"); r["b_cot_v1"] = r.get("b_cot"); r["faith_v1"] = r.get("faith")
            r["b_int"] = bis; r["b_cot"] = bcs; r["faith"] = faith
            r["static_label"] = static; r["n_cpos"] = len(votes)
            r["reward"] = [float(acc) + a.lambda_faith * fa for acc, fa in zip(r["acc"], faith)]
            r["scoring"] = "S-HINT-EVAL-v2 (probe v2 C-pos on base + v3 judge) per rollout; static = majority"
            cc.update((bi, bc) for bi, bc in zip(bis, bcs) if bi is not None and bc is not None)
            f.write(json.dumps(r) + "\n")
            r2 = dict(r); r2["acc"] = [1.0] * K; r2["reward"] = [float(fa) for fa in faith]
            f2.write(json.dumps(r2) + "\n")
    n = sum(cc.values())
    print("rollout cell distribution (C-pos rows):", {str(k): round(v / n, 3) for k, v in sorted(cc.items())},
          "| faith rate", round(sum(v for k, v in cc.items() if k[0] == k[1]) / n, 3),
          "| no-C-pos rollouts", sum(x is None for x in b_int), "/", len(b_int),
          "| prompts without any C-pos", n_nocpos_prompt, flush=True)
    sl = Counter(static_labels.values()); print("static label (majority) distribution:", dict(sl), flush=True)
    print("saved", out_path, "and", out_path.replace(".jsonl", "_faithonly.jsonl"), flush=True)

    if a.static_dataset:
        dd = load_from_disk(a.dataset)
        tr = dd["train"].filter(lambda ex: int(ex["index"]) in static_labels)
        tr = tr.map(lambda ex: {"labels": int(static_labels[int(ex["index"])]), "labels_v1_static": int(ex["labels"])})
        new = DatasetDict({"train": tr, **{sp: dd[sp] for sp in dd if sp != "train"}})
        new.save_to_disk(a.static_dataset)
        agree = sum(int(ex["labels"]) == int(ex["labels_v1_static"]) for ex in tr) / max(len(tr), 1)
        print(f"static dataset saved to {a.static_dataset}: train {len(tr)} rows, labels = majority static "
              f"(agreement with the old v1 column {agree:.3f})", flush=True)


if __name__ == "__main__":
    main()
