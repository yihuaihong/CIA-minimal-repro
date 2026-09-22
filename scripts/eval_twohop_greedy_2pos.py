"""TwoHopFact greedy eval — NEW probe + 2-position UNION + multi top-K.

For each test prompt:
  1. Greedy generation (CoT) with chat-formatted 2-hop prompt
  2. Position 1: hidden at last token of e1 in CHAT-FORMATTED 2-hop prompt → probe → top-K
  3. Position 2: hidden at last token of e1 inside (chat + CoT first step) → probe → top-K
  4. B_INT@K = (bridge_first_token ∈ topK_p1) OR (∈ topK_p2)  per paper §C.1
  5. B_CoT = first-word match of pred_bridge from CoT step-1

Output: 4-cell breakdown + CIA per (top-K ∈ {50, 100, 200}).
"""
import argparse, os, re, json, sys
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from tqdm import tqdm

# Reuse existing bridge extractor (handles markdown bold, quotes, descriptive
# prefixes, role descriptions). See CPF_utils/evaluation_utils.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from CPF_utils.evaluation_utils import extract_bridge_entity


PROMPT_TEMPLATE = """You are an expert in multi-hop factual reasoning. For the following question, you must reason step by step using exactly two hops. Always explicitly identify and state the intermediate entity (the 'bridge entity') before giving the final answer. Structure your response exactly as follows:

1. First, identify the bridge entity by applying the first implied relation.
2. Then, apply the second relation to the bridge entity to find the final answer.
3. Finally, state the complete answer with the prefix 'FINAL ANSWER:'.

Example:
Question: The mother of the spouse of Hailey Bieber is named

1. The spouse of Hailey Bieber is Justin Bieber (bridge entity).
2. The mother of Justin Bieber is Pattie Mallette.
FINAL ANSWER: Pattie Mallette

Now answer the following question in exactly the same structured format (steps 1-3, explicitly state the bridge entity):
{question}"""


STEP1_RE = re.compile(r"(?ms)^\s*1\.\s*(.*?)(?=^\s*2\.|FINAL ANSWER:)")


def first_word(s: str) -> str:
    if not s: return ""
    toks = s.strip().split()
    return toks[0].lower().strip("*\"',.;:!?()[]") if toks else ""


def extract_step1(full_gen: str) -> str:
    m = STEP1_RE.search(full_gen)
    if m: return m.group(1).strip()
    return "\n".join(full_gen.splitlines()[:2]).strip()


def extract_pred_bridge(full_gen: str) -> str:
    """Use existing CPF_utils.evaluation_utils.extract_bridge_entity — handles
    markdown bold, quotes, descriptive prefixes, role-based extractions."""
    result = extract_bridge_entity(full_gen)
    return result or ""


def find_last_e1_token(tokenizer, input_ids_1d, attn_mask_1d, e1_str: str) -> int:
    """Last token containing last char of last occurrence of e1_str. -1 if not found."""
    if not e1_str: return -1
    n_valid = int(attn_mask_1d.sum().item())
    ids = input_ids_1d[:n_valid].tolist()
    pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in ids]
    full = "".join(pieces)
    target = e1_str.strip()
    pos = full.lower().rfind(target.lower())
    if pos < 0: return -1
    end_char = pos + len(target) - 1
    cursor = 0
    for i, p in enumerate(pieces):
        next_cursor = cursor + len(p)
        if end_char < next_cursor: return i
        cursor = next_cursor
    return n_valid - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF model path (base or trained ckpt)")
    ap.add_argument("--base_name", required=True, help="Qwen3-8B / gemma-2-9b-it / Llama-3.1-8B-Instruct")
    ap.add_argument("--probe_path", required=True, help="Path to probe_chat_filtered_<model>.pt")
    ap.add_argument("--csv_path",
                    default=f"{os.environ['SCRATCH']}/datasets/TwoHopFact/TwoHopFact.csv")
    ap.add_argument("--split_fraction", nargs=3, type=float, default=[0.6, 0.2, 0.2],
                    help="train/val/test fractions; eval runs on test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=0, help="0 = full test split")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", default=None)
    ap.add_argument("--top_ks", nargs="+", type=int, default=[50, 100, 200])
    args = ap.parse_args()

    print(f"═══ TwoHop greedy 2-pos eval: {args.base_name} ═══", flush=True)
    print(f"  model_path: {args.model_path}", flush=True)
    print(f"  probe: {args.probe_path}", flush=True)

    # ── Load tokenizer + model ──
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if "qwen3" in args.base_name.lower():
        _orig = tok.apply_chat_template
        def _no_think(*a, **k): k.setdefault("enable_thinking", False); return _orig(*a, **k)
        tok.apply_chat_template = _no_think
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16
    ).cuda().eval()

    # ── Load probe ──
    probe_ckpt = torch.load(args.probe_path, map_location="cpu", weights_only=False)
    hidden_dim = probe_ckpt["hidden_dim"]
    vocab_size = probe_ckpt["vocab_size"]
    layer = probe_ckpt["layer"]
    probe = nn.Linear(hidden_dim, vocab_size)
    probe.load_state_dict(probe_ckpt["probe_state_dict"])
    probe = probe.cuda().eval()
    print(f"  probe layer={layer}, hidden_dim={hidden_dim}, vocab_size={vocab_size}", flush=True)

    # ── Load CSV + reproduce same shuffle + take TEST split ──
    df = pd.read_csv(args.csv_path)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_total = len(df)
    n_tr = int(n_total * args.split_fraction[0])
    n_va = int(n_total * args.split_fraction[1])
    test_df = df.iloc[n_tr + n_va:].reset_index(drop=True)
    if args.n_samples > 0 and args.n_samples < len(test_df):
        test_df = test_df.iloc[:args.n_samples].reset_index(drop=True)
    n = len(test_df)
    print(f"  test split: n={n}", flush=True)

    e1_prompts = test_df["r1(e1).prompt"].astype(str).tolist()
    e1_values = test_df["e1.value"].astype(str).tolist()
    e2_values = test_df["e2.value"].astype(str).tolist()
    two_hop_questions = test_df["r2(r1(e1)).prompt"].astype(str).tolist()
    # Gold FINAL ANSWER = e3.value (+ e3.aliases) for accuracy.
    e3_values = test_df["e3.value"].astype(str).tolist()
    import ast as _ast
    def _parse_aliases(s, fallback):
        try:
            t = _ast.literal_eval(s)
            out = []
            for grp in t:
                out.extend(str(x) for x in grp)
            return out or [fallback]
        except Exception:
            return [fallback]
    e3_aliases = [_parse_aliases(a, v) for a, v in zip(test_df["e3.aliases"].astype(str).tolist(), e3_values)]

    # Bridge first-token ids
    bridge_tids = []
    for b in e2_values:
        ids = tok.encode(b.strip(), add_special_tokens=False)
        bridge_tids.append(ids[0] if ids else -1)

    # ── Eval loop ──
    writer = jsonlines.open(args.out_jsonl, "w") if args.out_jsonl else None
    n_p1_in_topk = {k: 0 for k in args.top_ks}
    n_p2_in_topk = {k: 0 for k in args.top_ks}
    n_union_in_topk = {k: 0 for k in args.top_ks}
    n_b_cot = 0
    n_acc = 0
    # accuracy stratified by the K=max cell, to see where acc sits
    acc_by_cell = {k: {(b, c): [0, 0] for b in [0, 1] for c in [0, 1]} for k in args.top_ks}
    n_cells = {k: {(b, c): 0 for b in [0, 1] for c in [0, 1]} for k in args.top_ks}
    _FINAL_RE = re.compile(r"FINAL ANSWER:?\s*(.+)", re.IGNORECASE)

    def _norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    def final_answer_correct(gen_text, gold, aliases):
        m = _FINAL_RE.search(gen_text)
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

    for i in tqdm(range(n)):
        q_2hop = two_hop_questions[i]
        e1 = e1_values[i]
        bridge_tid = bridge_tids[i]
        bridge_first_word = first_word(e2_values[i])

        # ── Chat-formatted 2-hop prompt ──
        user_content = PROMPT_TEMPLATE.format(question=q_2hop)
        msg = [{"role": "user", "content": user_content}]
        chat = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

        # ── Greedy generate CoT ──
        enc = tok(chat, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

        # ── B_CoT: first BPE token match of pred_bridge ──
        # (matches existing cpf_evaluation pipeline: cot_token_id == correct_token_id)
        pred_bridge = extract_pred_bridge(gen)
        if pred_bridge:
            pred_tids = tok.encode(pred_bridge.strip(), add_special_tokens=False)
            pred_tid = pred_tids[0] if pred_tids else -1
        else:
            pred_tid = -1
        b_cot = int(pred_tid >= 0 and pred_tid == bridge_tid)

        # ── Position 1: forward chat, probe at last e1 token ──
        enc1 = tok(chat, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        pos1 = find_last_e1_token(tok, enc1.input_ids[0].cpu(), enc1.attention_mask[0].cpu(), e1)
        topks_p1 = {k: set() for k in args.top_ks}
        if pos1 >= 0:
            with torch.no_grad():
                o1 = model(**enc1, output_hidden_states=True, use_cache=False)
            h1 = o1.hidden_states[layer + 1][0, pos1, :].float()
            logits1 = probe(h1)
            for k in args.top_ks:
                topks_p1[k] = set(logits1.topk(k).indices.cpu().tolist())

        # ── Position 2: forward (chat + step1), probe at last e1 in step ──
        step1 = extract_step1(gen)
        if step1:
            chat_dual = chat + "\n" + step1
            enc2 = tok(chat_dual, return_tensors="pt", truncation=True, max_length=768).to("cuda")
            pos2 = find_last_e1_token(tok, enc2.input_ids[0].cpu(), enc2.attention_mask[0].cpu(), e1)
            topks_p2 = {k: set() for k in args.top_ks}
            if pos2 >= 0:
                with torch.no_grad():
                    o2 = model(**enc2, output_hidden_states=True, use_cache=False)
                h2 = o2.hidden_states[layer + 1][0, pos2, :].float()
                logits2 = probe(h2)
                for k in args.top_ks:
                    topks_p2[k] = set(logits2.topk(k).indices.cpu().tolist())
        else:
            pos2 = -1
            topks_p2 = {k: set() for k in args.top_ks}

        # ── Aggregate ──
        n_b_cot += b_cot
        acc = final_answer_correct(gen, e3_values[i], e3_aliases[i])
        n_acc += acc
        per_sample = {"i": i, "e1": e1, "e2": e2_values[i], "pred_bridge": pred_bridge,
                       "b_cot": b_cot, "acc": acc, "final_gold": e3_values[i], "step1": step1[:200]}
        for k in args.top_ks:
            p1 = int(bridge_tid >= 0 and bridge_tid in topks_p1[k])
            p2 = int(bridge_tid >= 0 and bridge_tid in topks_p2[k])
            union = int(p1 or p2)
            n_p1_in_topk[k] += p1
            n_p2_in_topk[k] += p2
            n_union_in_topk[k] += union
            n_cells[k][(union, b_cot)] += 1
            acc_by_cell[k][(union, b_cot)][0] += acc
            acc_by_cell[k][(union, b_cot)][1] += 1
            per_sample[f"b_int@{k}"] = union
            per_sample[f"p1@{k}"] = p1
            per_sample[f"p2@{k}"] = p2
        if writer: writer.write(per_sample)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{n}] union@200={n_union_in_topk[200]/(i+1):.3f}  b_cot={n_b_cot/(i+1):.3f}", flush=True)

    if writer: writer.close()

    # ── Summary ──
    summary = {
        "base_name": args.base_name, "model_path": args.model_path,
        "probe_path": args.probe_path, "methodology": "GREEDY 2-pos UNION",
        "n_samples": n, "B_CoT_rate": n_b_cot / n,
        "accuracy": n_acc / n,
        "per_top_k": {},
    }
    print("\n══ Results (TwoHop test split, NEW probe, 2-pos UNION, greedy) ══", flush=True)
    print(f"  n={n}, B_CoT rate (first-word match): {n_b_cot/n:.4f}", flush=True)
    print(f"  ACCURACY (FINAL ANSWER vs e3.value+aliases): {n_acc/n:.4f}", flush=True)
    for k in args.top_ks:
        cia = (n_cells[k][(1, 1)] + n_cells[k][(0, 0)]) / n
        summary["per_top_k"][f"k={k}"] = {
            "B_INT_rate_p1": n_p1_in_topk[k] / n,
            "B_INT_rate_p2": n_p2_in_topk[k] / n,
            "B_INT_rate_union": n_union_in_topk[k] / n,
            "CIA": cia,
            "pct_11": n_cells[k][(1, 1)] / n * 100,
            "pct_10": n_cells[k][(1, 0)] / n * 100,
            "pct_01": n_cells[k][(0, 1)] / n * 100,
            "pct_00": n_cells[k][(0, 0)] / n * 100,
            "acc_by_cell": {f"{b}{c}": (acc_by_cell[k][(b, c)][0] / acc_by_cell[k][(b, c)][1]
                                        if acc_by_cell[k][(b, c)][1] else None)
                            for b in [0, 1] for c in [0, 1]},
        }
        print(f"  K={k:3d}: B_INT(p1)={n_p1_in_topk[k]/n:.4f}  B_INT(p2)={n_p2_in_topk[k]/n:.4f}  "
              f"UNION={n_union_in_topk[k]/n:.4f}  CIA={cia:.4f}  "
              f"(1,1)={summary['per_top_k'][f'k={k}']['pct_11']:.2f}  "
              f"(0,0)={summary['per_top_k'][f'k={k}']['pct_00']:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(summary, open(args.out_json, "w"), indent=2)
    print(f"  saved: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
