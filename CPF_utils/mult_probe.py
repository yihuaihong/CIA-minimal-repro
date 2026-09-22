"""Multiplication-task linear probe (paper §C.3).

Input:
    A corruption-labeled jsonl produced by
    `CPF_utils/multiplication_corruption.py`, where each row has:
        - approach              : 'A' or 'B' (parsed from original CoT)
        - follows_partial_products  : bool  ← B_INT ground-truth
        - full_generation       : CoT text (needed to locate summation)
        - corruption_details    : per-PP intervention results

Feature:
    Hidden state at the token position **immediately preceding the
    generation of the summation result**. We re-run model forward over
    `prompt + CoT_up_to_summation` and grab the last token's hidden
    state at the Table-7 probe layer.

Probe:
    nn.Linear(hidden, 2) trained with CE on B_INT labels. Standardize
    features, grad-clip, early-stop on val acc — mirror hint_probe.py.

CLI:
    python -m CPF_utils.mult_probe train_and_label \
        --labeled_jsonl <corruption_labeled.jsonl> \
        --model_name Meta-Llama-3-8B-Instruct --seed 8888

Writes a `*_probe_labeled.jsonl` alongside input with a new
`probe_internal` field per row — ready to be consumed by
`multiplication_parametric_faithfulness_reward` in probe mode.
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Sequence

import jsonlines
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPF_utils.layer_config import get_best_layer


# Matches the summation number (result of adding partial products).
# In paper/repo CoT format it usually appears after the 2nd PP block and
# before "FINAL ANSWER:". We split at that point to recover the "prompt
# up to the summation" prefix.
_PP_RE = re.compile(r"^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)",
                    re.MULTILINE)
_FINAL_RE = re.compile(r"FINAL ANSWER:?\s*(\d+)", re.IGNORECASE)


def locate_pre_summation_prefix(full_generation: str) -> str | None:
    """Return everything up to (but not including) the first line that
    contains the summation result. If we can't find ≥2 PPs we fall back
    to the entire generation minus the final-answer line.
    """
    pps = list(_PP_RE.finditer(full_generation))
    if len(pps) < 2:
        # Fallback: cut before FINAL ANSWER.
        fa = _FINAL_RE.search(full_generation)
        return full_generation[:fa.start()].rstrip() if fa else full_generation.rstrip()

    # Cut just after the second PP line (end of line).
    second_pp_end = pps[1].end()
    nl_idx = full_generation.find("\n", second_pp_end)
    cut = nl_idx + 1 if nl_idx >= 0 else second_pp_end
    # Advance past any "------" separator block that appears before the
    # summation number.
    rest = full_generation[cut:]
    sep_iter = re.finditer(r"^-{3,}\s*$", rest, re.MULTILINE)
    last_sep_end = None
    for m in sep_iter:
        # Stop once we're past the summation line (find first standalone digits after PPs).
        if re.search(r"^\s*\d{3,9}\s*$", rest[:m.start()], re.MULTILINE):
            break
        last_sep_end = m.end()
    if last_sep_end is not None:
        cut = cut + last_sep_end
    return full_generation[:cut].rstrip()


def locate_pre_answer_prefix(full_generation: str) -> str | None:
    """Uniform probe position across Approach-A and Approach-B.

    Returns the prefix of `full_generation` up to (but not including) the
    first digit of the FINAL ANSWER. The last token of this prefix is the
    model's hidden state immediately before it commits to emitting the
    numerical answer — semantically analogous whether the model wrote a
    long-mult derivation (B) or jumped directly to the answer (A).
    """
    fa = _FINAL_RE.search(full_generation)
    if fa is not None:
        return full_generation[: fa.start(1)]  # cut right before the digit group
    # Fallback: no explicit FINAL ANSWER marker. Keep the whole generation.
    return full_generation.rstrip()


def derive_b_int_label(record: dict, mode: str) -> int | None:
    """Derive a binary B_INT label from a corruption-labeled record.

    Modes:
      - "tracked"       : paper label. +1 iff at least one pp corruption
                          caused the model's new_sum to EXACTLY match
                          (corrupted_val + other_val). Strict.
      - "not_recalled"  : user-proposed label. +1 iff at least one pp
                          corruption caused the model's new_sum to differ
                          from the original sum (i.e., the corruption
                          `changed the output`). Loose.
    Returns None if the record has no corruption_details to label from.
    """
    cd = record.get("corruption_details") or {}
    if not cd or not isinstance(cd, dict):
        return None
    # Only Approach-B records have corruption-based labels.
    if record.get("approach") != "B":
        return None
    if mode == "tracked":
        hit = any(bool(cd.get(pp, {}).get("tracked", False)) for pp in ("pp1", "pp2"))
        return int(hit)
    elif mode == "not_recalled":
        # Positive when at least one pp corruption perturbed the output.
        any_pp = False
        any_not_recalled = False
        for pp in ("pp1", "pp2"):
            d = cd.get(pp)
            if not d:
                continue
            any_pp = True
            if not bool(d.get("recalled", False)):
                any_not_recalled = True
                break
        return int(any_not_recalled) if any_pp else None
    else:
        raise ValueError(f"unknown label_mode: {mode}")


def make_mult_chat_prefix(tokenizer, raw_question: str, cot_prefix: str) -> str:
    """Apply the same chat template GRPO/eval use, then append the
    CoT-up-to-summation prefix as the start of the assistant's reply."""
    from CPF_utils.chat import patch_tokenizer, user_chat
    patch_tokenizer(tokenizer, getattr(tokenizer, "name_or_path", ""))
    return user_chat(tokenizer, raw_question) + cot_prefix


@torch.no_grad()
def extract_hidden_at_last_token(
        model, tokenizer, texts: Sequence[str],
        layer: int, batch_size: int = 4,
        max_length: int = 2048) -> torch.Tensor:
    """Last-real-token hidden state at `layer` (0-indexed transformer
    block). Uses right padding to avoid the Llama-3 bf16 + left-pad NaN
    issue encountered in hint_probe.py."""
    feats = []
    device = model.device
    tokenizer.padding_side = "right"
    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"hs layer {layer}"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer + 1]
        # Right padding: last non-pad index = sum(attention_mask)-1.
        last_idx = inputs.attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        feats.append(hs.gather(1, idx).squeeze(1).float().cpu())
    return torch.cat(feats, dim=0)


def train_probe(X_train, y_train, X_val, y_val,
                epochs: int = 12, lr: float = 5e-4,
                weight_decay: float = 0.01, batch_size: int = 64,
                device: str = "cuda") -> tuple[nn.Linear, dict]:
    hidden = X_train.shape[1]
    probe = nn.Linear(hidden, 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                            weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    Xtr, ytr = X_train.to(device), y_train.to(device)
    Xv, yv = X_val.to(device), y_val.to(device)

    best_state, best_acc = None, -1.0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for s in range(0, len(Xtr), batch_size):
            bs = perm[s:s + batch_size]
            logits = probe(Xtr[bs])
            loss = ce(logits, ytr[bs])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(bs)
        probe.eval()
        with torch.no_grad():
            acc = (probe(Xv).argmax(-1) == yv).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
        print(f"  ep {ep+1}/{epochs}  loss={total/len(Xtr):.4f}  val_acc={acc:.4f}")
    probe.load_state_dict(best_state)
    return probe, {"val_acc": best_acc, "epochs": epochs}


def train_and_label(
        labeled_jsonl: str, model_name: str, model_dir: str,
        seed: int = 8888, train_frac: float = 0.6, val_frac: float = 0.2,
        device: int = 0, dtype: str = "bfloat16",
        output: str | None = None,
        label_mode: str = "tracked",
        probe_position: str = "pre_answer",
        n_bootstrap: int = 1,
        layer_override: int | None = None) -> str:
    """Train a B_INT probe on corruption-labeled data and apply to ALL rows.

    Args:
        label_mode: "tracked" (paper §C.3) or "not_recalled" (user proposal,
            positive iff corruption changed the output).
        probe_position: "pre_answer" (uniform across A/B; cuts before the
            FINAL ANSWER digit) or "pre_summation" (paper; only works for
            B-samples that contain explicit partial-product block).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)

    records = list(jsonlines.open(labeled_jsonl))
    print(f"Loaded {len(records)} corruption-labeled records")
    print(f"Label mode: {label_mode}  |  Probe position: {probe_position}")

    # Training set = rows with a valid B_INT label. derive_b_int_label returns
    # None for Approach-A samples, so these end up in the "apply-only" pool.
    labeled_rows = []
    for r in records:
        y = derive_b_int_label(r, label_mode)
        if y is not None:
            labeled_rows.append((r, y))
    print(f"Labelled rows for training: {len(labeled_rows)}")
    n_pos_lbl = sum(y for _, y in labeled_rows)
    print(f"  positives: {n_pos_lbl}  negatives: {len(labeled_rows)-n_pos_lbl}")
    if len(labeled_rows) < 50:
        raise RuntimeError("Not enough labelled samples for probe training; "
                           "re-run multiplication_corruption.py on a larger set.")
    if n_pos_lbl < 5 or (len(labeled_rows) - n_pos_lbl) < 5:
        raise RuntimeError(f"Severe label imbalance under mode={label_mode}: "
                           f"pos={n_pos_lbl}, neg={len(labeled_rows)-n_pos_lbl}. "
                           "Switch label_mode.")

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, model_name),
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, model_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    if layer_override is not None:
        layer = layer_override
        print(f"Using probe layer {layer}/{n_layers-1} (override)")
    else:
        layer = get_best_layer(model_name, "multiplication", n_layers=n_layers)
        print(f"Using probe layer {layer}/{n_layers-1} (Table 7)")

    # ---- Build prefix strings for ALL records (training subset + A samples).
    def build_prefix(r: dict) -> str:
        if probe_position == "pre_answer":
            pref = locate_pre_answer_prefix(r.get("full_generation", ""))
        else:
            pref = locate_pre_summation_prefix(r.get("full_generation", ""))
        return make_mult_chat_prefix(tokenizer, r.get("prompt", ""), pref or "")

    all_texts = [build_prefix(r) for r in records]
    print(f"Built {len(all_texts)} prompt+CoT strings (all records)")

    # ---- Extract hidden states for ALL records.
    X_all = extract_hidden_at_last_token(model, tokenizer, all_texts, layer,
                                          batch_size=4)
    bad = torch.isnan(X_all).any(dim=1) | torch.isinf(X_all).any(dim=1)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"WARN: {n_bad} bad feature rows; zeroing them out")
    X_clean = X_all[~bad]
    if len(X_clean) == 0:
        raise RuntimeError("All feature rows are NaN/Inf")
    mu = X_clean.mean(0, keepdim=True)
    sigma = X_clean.std(0, keepdim=True).clamp_min(1e-6)
    print(f"  clean stats: |X|max={X_clean.abs().max():.3f} "
          f"mean={X_clean.mean():.4f} std={X_clean.std():.4f}")
    X_all = torch.where(bad.unsqueeze(1), torch.zeros_like(X_all),
                        (X_all - mu) / sigma)

    # ---- Map records → row index in X_all (preserves record order).
    rec_idx = {id(r): i for i, r in enumerate(records)}
    # Training subset indices + labels.
    tr_indices = np.array([rec_idx[id(r)] for r, _ in labeled_rows])
    tr_labels  = torch.tensor([y for _, y in labeled_rows], dtype=torch.long)
    # Drop NaN rows from training pool.
    good_mask = ~bad[tr_indices]
    tr_indices = tr_indices[good_mask.numpy()]
    tr_labels  = tr_labels[good_mask]
    print(f"Usable (non-NaN) labelled rows: {len(tr_indices)}")

    # ---- Bootstrap over K splits for robust train/val/test stats.
    train_accs, val_accs, test_accs, test_f1s = [], [], [], []
    best_val, best_probe, best_mu, best_sigma = -1.0, None, mu, sigma
    for b in range(max(1, n_bootstrap)):
        b_seed = seed + b
        rng = np.random.default_rng(b_seed)
        perm = rng.permutation(len(tr_indices))
        n_tr  = int(len(perm) * train_frac)
        n_val = int(len(perm) * val_frac)
        tr_sel = perm[:n_tr]
        va_sel = perm[n_tr:n_tr + n_val]
        te_sel = perm[n_tr + n_val:]
        X_tr = X_all[tr_indices[tr_sel]]; y_tr = tr_labels[tr_sel]
        X_va = X_all[tr_indices[va_sel]]; y_va = tr_labels[va_sel]
        probe_b, info_b = train_probe(X_tr, y_tr, X_va, y_va,
                                      epochs=12, lr=5e-4, weight_decay=0.01)
        val_accs.append(info_b["val_acc"])

        # Train accuracy (same split the probe learned on — diagnostic).
        with torch.no_grad():
            X_tr_dev = X_tr.to("cuda"); y_tr_dev = y_tr.to("cuda")
            preds_tr = probe_b(X_tr_dev).argmax(-1)
            acc_tr = (preds_tr == y_tr_dev).float().mean().item()
        train_accs.append(acc_tr)

        if len(te_sel) > 0:
            X_te = X_all[tr_indices[te_sel]].to("cuda")
            y_te = tr_labels[te_sel].to("cuda")
            with torch.no_grad():
                logits_te = probe_b(X_te)
                preds_te  = logits_te.argmax(-1)
                acc_te    = (preds_te == y_te).float().mean().item()
            test_accs.append(acc_te)
            tp = ((preds_te == 1) & (y_te == 1)).sum().item()
            fp = ((preds_te == 1) & (y_te == 0)).sum().item()
            fn = ((preds_te == 0) & (y_te == 1)).sum().item()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            test_f1s.append(f1)

        # Keep best-by-val probe for application.
        if info_b["val_acc"] > best_val:
            best_val = info_b["val_acc"]
            best_probe = probe_b
        print(f"  boot {b+1}/{n_bootstrap}: train_acc={acc_tr:.4f}  "
              f"val_acc={info_b['val_acc']:.4f}  "
              f"test_acc={test_accs[-1] if test_accs else float('nan'):.4f}  "
              f"pos_F1={test_f1s[-1] if test_f1s else float('nan'):.4f}")

    probe = best_probe
    info = {"val_acc": best_val,
            "train_acc_mean": float(np.mean(train_accs)),
            "train_acc_std":  float(np.std(train_accs)),
            "val_acc_mean": float(np.mean(val_accs)),
            "val_acc_std":  float(np.std(val_accs)),
            "test_acc_mean": float(np.mean(test_accs)) if test_accs else None,
            "test_acc_std":  float(np.std(test_accs)) if test_accs else None,
            "test_f1_mean":  float(np.mean(test_f1s))  if test_f1s  else None,
            "test_f1_std":   float(np.std(test_f1s))   if test_f1s  else None,
            "n_bootstrap": len(val_accs)}
    print(f"\n== Bootstrap summary (k={len(val_accs)}) ==")
    print(f"  train_acc:{info['train_acc_mean']:.4f} ± {info['train_acc_std']:.4f}")
    print(f"  val_acc:  {info['val_acc_mean']:.4f} ± {info['val_acc_std']:.4f}  (best={best_val:.4f})")
    if test_accs:
        print(f"  test_acc: {info['test_acc_mean']:.4f} ± {info['test_acc_std']:.4f}")
        print(f"  pos_F1:   {info['test_f1_mean']:.4f} ± {info['test_f1_std']:.4f}")

    # ---- Apply probe to ALL records.
    probe.eval()
    with torch.no_grad():
        logits_all = probe(X_all.to("cuda"))
        probs_all  = torch.softmax(logits_all, dim=-1)[:, 1].cpu().tolist()
        preds_all  = logits_all.argmax(-1).cpu().tolist()

    # ---- Write output jsonl: every record gets probe_internal + probe_prob.
    out_path = output or Path(labeled_jsonl).with_name(
        Path(labeled_jsonl).stem + f"_probe_{label_mode}.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(out_path), "w") as w:
        for r, pred, prob in zip(records, preds_all, probs_all):
            r = dict(r)
            r["probe_internal"] = bool(pred)
            r["probe_prob"]     = float(prob)
            r["probe_label_mode"] = label_mode
            r["probe_position"] = probe_position
            w.write(r)
    print(f"Wrote {out_path}")

    # ---- Distributional summary: probe scores by (model, declared_approach).
    from collections import defaultdict
    buckets = defaultdict(list)
    for r, prob in zip(records, probs_all):
        buckets[r.get("approach") or "UNK"].append(prob)
    print("--- Probe score distribution by declared approach ---")
    for k in sorted(buckets):
        arr = np.array(buckets[k])
        if len(arr) == 0:
            continue
        print(f"  approach={k}  n={len(arr)}  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"P(>0.5)={(arr > 0.5).mean():.3f}")

    # ---- Save probe weights.
    ckpt_path = Path(out_path).with_suffix(".probe.pt")
    torch.save({
        "state_dict": probe.state_dict(),
        "layer": layer,
        "mean": mu.squeeze(0).cpu(),
        "std": sigma.squeeze(0).cpu(),
        "model_name": model_name,
        "label_mode": label_mode,
        "probe_position": probe_position,
        "val_acc": info["val_acc"],
        "val_acc_mean": info.get("val_acc_mean"),
        "val_acc_std":  info.get("val_acc_std"),
        "test_acc_mean": info.get("test_acc_mean"),
        "test_acc_std":  info.get("test_acc_std"),
        "test_f1_mean":  info.get("test_f1_mean"),
        "test_f1_std":   info.get("test_f1_std"),
        "n_bootstrap":   info.get("n_bootstrap", 1),
    }, ckpt_path)
    print(f"Saved probe checkpoint to {ckpt_path}")

    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train_and_label")
    p.add_argument("--labeled_jsonl", required=True,
                   help="output of multiplication_corruption.py")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_dir", default="${SCRATCH}/transformers")
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output", default=None)
    p.add_argument("--label_mode", choices=["tracked", "not_recalled"],
                   default="not_recalled",
                   help="tracked = paper §C.3 strict label; "
                        "not_recalled = positive iff corruption changed output")
    p.add_argument("--probe_position", choices=["pre_answer", "pre_summation"],
                   default="pre_answer",
                   help="pre_answer: uniform A/B cut before FINAL ANSWER digit; "
                        "pre_summation: paper's pre-sum position (B-only)")
    p.add_argument("--n_bootstrap", type=int, default=1,
                   help="Number of train/val splits to aggregate stats over "
                        "(default 1 = single split).")
    p.add_argument("--layer", type=int, default=None,
                   help="Override probe layer (skip Table 7 lookup).")
    args = ap.parse_args()

    if args.cmd == "train_and_label":
        train_and_label(
            labeled_jsonl=args.labeled_jsonl, model_name=args.model_name,
            model_dir=args.model_dir, seed=args.seed,
            train_frac=args.train_frac, val_frac=args.val_frac,
            device=args.device, dtype=args.dtype, output=args.output,
            label_mode=args.label_mode, probe_position=args.probe_position,
            n_bootstrap=args.n_bootstrap,
            layer_override=args.layer)


if __name__ == "__main__":
    main()
