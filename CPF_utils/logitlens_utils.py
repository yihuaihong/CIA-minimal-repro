import torch
from tqdm import tqdm
import transformers
from CPF_utils import tokenization_utils
from CPF_utils.layer_config import get_best_layer
import jsonlines
import os
import re
import numpy as np


def _safe_fallback_tid(tokenizer) -> int:
    """Return a guaranteed non-None token id. Llama-3's tokenizer has
    unk_token_id=None, which poisons downstream `torch.tensor(ids)`.
    Fall back to pad/eos/0 in that order."""
    for attr in ("unk_token_id", "pad_token_id", "eos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        if isinstance(tid, (list, tuple)):
            tid = tid[0] if tid else None
        if tid is not None:
            return int(tid)
    return 0


# ================== 工具函数 ==================

def get_hidden_states(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Get hidden states from the model for the given prompt inputs.

    Args:
        model: The model to get hidden states from.
        prompt_inputs: The prompt inputs.

    Returns:
        The hidden states as a torch.Tensor of shape [n_layers, B, seq_len, hidden].
    """
    with torch.no_grad():
        outputs = model(**prompt_inputs, output_hidden_states=True)

    # outputs.hidden_states 是 tuple，长度 = n_layers + 1（含 embedding 层）
    # 每个元素 shape: [B, seq_len, hidden]
    hidden_states = torch.stack(outputs.hidden_states[1:])  # [n_layers, B, seq_len, hidden]
    return hidden_states


def check_topk(values_map: torch.Tensor, key_token_ids: list, k: int = 100):
    """
    :param values_map: [batch, vocab]，在 GPU 上
    :param key_token_ids: List[int]，长度 == batch_size
    :param k: top-k 范围
    :return:
        in_topk_list: List[bool]
        rank_list:    List[int]，1-indexed
        top1_list:    List[int]，每个样本 top-1 的 token_id
    """
    batch_size = values_map.size(0)
    assert len(key_token_ids) == batch_size, "key_token_ids 长度必须与 batch_size 一致"

    # key_token_ids 转为 GPU tensor，避免逐样本 Python 循环
    key_ids = torch.tensor(key_token_ids, device=values_map.device)  # [B]

    sorted_indices = torch.argsort(values_map, dim=1, descending=True)  # [B, vocab]

    # ── in_topk：向量化判断 ───────────────────────────────────────────────
    topk_indices = sorted_indices[:, :k]                                # [B, k]
    in_topk = (topk_indices == key_ids.unsqueeze(1)).any(dim=1)         # [B]

    # ── rank：向量化计算 ──────────────────────────────────────────────────
    match = (sorted_indices == key_ids.unsqueeze(1))                    # [B, vocab]
    # argmax 找第一个 True 的位置，若全为 False 则返回 0（需特殊处理）
    ranks = match.float().argmax(dim=1) + 1                             # [B]，1-indexed
    # 对于未命中的样本，rank 设为 vocab_size
    vocab_size = values_map.size(1)
    ranks = torch.where(match.any(dim=1), ranks,
                        torch.tensor(vocab_size, device=values_map.device))

    # ── top1：每个样本 top-1 token id ────────────────────────────────────
    top1 = sorted_indices[:, 0]                                         # [B]

    # 最后统一转 CPU，只做一次设备转移
    return in_topk.cpu().tolist(), ranks.cpu().tolist(), top1.cpu().tolist()


def merge_results(results_list: list):
    """
    跨层合并：
      - in_topk: 任意层命中则为 True
      - rank:    取各层最小 rank
      - top1:    取 rank 最小那层的 top-1 token_id

    :param results_list: List[ (in_topk_list, rank_list, top1_list) ]  长度 = num_layers
    :return: merged_in_topk, merged_ranks, merged_top1
    """
    batch_size = len(results_list[0][0])
    merged_in_topk = [False] * batch_size
    merged_ranks   = [float('inf')] * batch_size
    merged_top1    = [None] * batch_size

    for in_topk_list, rank_list, top1_list in results_list:
        for i in range(batch_size):
            merged_in_topk[i] = merged_in_topk[i] or in_topk_list[i]
            if rank_list[i] < merged_ranks[i]:
                merged_ranks[i] = rank_list[i]
                merged_top1[i]  = top1_list[i]

    return merged_in_topk, merged_ranks, merged_top1


def logit_lens(prompts, subject_prompts, model, tokenizer,
               source_layer_idxs, bridge_entities, top_k,
               cot_pred_bridges=None, all_tokens=False):
    """
    :param source_layer_idxs: 参与 merge 的层索引列表，None 则使用默认策略（前半段所有层）
    :return:
        merged_in_topk:      List[bool]
        merged_ranks:        List[int]
        merged_top1:         List[int]  跨层 best-rank 对应的 top-1 token_id
        cot_first_token_ids: List[int] | None
    """
    E = model.get_output_embeddings().weight.detach()  # 留在 GPU 上做矩阵乘法

    # ── 正确 bridge 的 first token id ─────────────────────────────────────
    correct_token_ids = []
    use_idx = 0 if 'qwen3' in model.config.model_type.lower() else 1
    for bridge in bridge_entities:
        tid = tokenizer(bridge).input_ids[use_idx]
        correct_token_ids.append(tid)

    # ── CoT pred bridge 的 first token id（可选）──────────────────────────
    _fb = _safe_fallback_tid(tokenizer)
    cot_first_token_ids = None
    if cot_pred_bridges is not None:
        cot_first_token_ids = []
        for bridge in cot_pred_bridges:
            if not bridge or bridge in ('nan', 'null'):
                cot_first_token_ids.append(_fb)
            else:
                tokens = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                cot_first_token_ids.append(tokens[0] if tokens else _fb)

    # ── 决定参与 merge 的层 ────────────────────────────────────────────────
    n_layers = model.config.num_hidden_layers

    if source_layer_idxs is None:
        # 策略一（默认）：前半段所有层
        # 32层模型 → [1, 2, ..., 16]
        source_layer_idxs = list(range(1, n_layers // 2 + 1))

        # 策略二：与 probing 的 best_layer 对齐（反注释以启用，同时注释掉策略一）
        # model_name = getattr(model.config, '_name_or_path', 'unknown_model')
        # if '/' in model_name:
        #     model_name = model_name.split('/')[-1]
        # key = (model_name, train_dataset_name)  # train_dataset_name 需作为参数传入
        # best_layer = best_layer_dict.get(key, n_layers // 2)
        # source_layer_idxs = [best_layer]

    # ── 前向传播 ──────────────────────────────────────────────────────────
    prompt_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    last_subject_token_positions = (
        tokenization_utils.find_exact_substrings_token_positions_from_tensor(
            tokenizer, prompt_inputs["input_ids"], subject_prompts
        )
    )

    results_list = []

    hidden_states = get_hidden_states(model, prompt_inputs)
    # shape: [n_layers, B, seq_len, hidden]，全程在 GPU 上

    if not all_tokens:
        # 从 seq_len 维取出每个样本 last subject token 位置的 hidden state
        t1_hidden_states = torch.stack(
            [hidden_states[:, i, pos, :]          # [n_layers, hidden]
             for i, pos in enumerate(last_subject_token_positions)],
            dim=1,
        )  # [n_layers, B, hidden]

        for i, t1_hidden_state in enumerate(t1_hidden_states):
            if i not in source_layer_idxs:
                continue
            values_map = t1_hidden_state.matmul(E.T)  # 保持在 GPU，check_topk 内部转 CPU
            layer_result = check_topk(values_map, correct_token_ids, k=top_k)
            results_list.append(layer_result)

    merged_in_topk, merged_ranks, merged_top1 = merge_results(results_list)
    return merged_in_topk, merged_ranks, merged_top1, cot_first_token_ids


def run_attn_lens_evaluation(model, dataset, dataset_name, tokenizer, batch_size=64, top_k=200):
    raise NotImplementedError(
        "attn_lens evaluation is not implemented. "
        "Use interp_tool='logit_lens' or 'linear_probe' instead."
    )


def _extract_first_cot_step(full_generation: str) -> str:
    """Return the text of the first CoT reasoning step.

    The prompt used by run_two_hop_acc_evaluation instructs the model to
    produce a response that starts with a numbered step "1. ...". We grab
    everything after "1." up to (but not including) "2." or
    "FINAL ANSWER:", whichever comes first. If neither pattern is found,
    fall back to the first two lines of the generation.
    """
    if not full_generation:
        return ""
    # Anchor on first "1." at line start.
    m = re.search(r"(?ms)^\s*1\.\s*(.*?)(?=^\s*2\.|FINAL ANSWER:)",
                  full_generation)
    if m:
        return m.group(1).strip()
    return "\n".join(full_generation.splitlines()[:2]).strip()


def run_logit_lens_evaluation(model, dataset, dataset_name,
                               eval_dataset_responses_path,
                               tokenizer, batch_size=64, top_k=200,
                               output_path=None, seed=8888,
                               probe_cot_first_step: bool = True):

    model_name = getattr(model.config, '_name_or_path', 'unknown_model')
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    interp_tool = "logit_lens"
    n_eval = len(dataset)

    print(f"Starting logit lens evaluation for model={model_name}, dataset={dataset_name} ...")

    fact_type = "r2(r1(e1))"
    eval_prompt_col  = f'{fact_type}.prompt'
    eval_subject_col = f'{fact_type}.subject_cut.prompt'
    eval_bridge_col  = 'e2.value'

    for col in [eval_prompt_col, eval_subject_col, eval_bridge_col]:
        if col not in dataset.columns:
            raise ValueError(f"dataset 缺少必要列: {col}")

    # ── 读取 CoT jsonl（可选）────────────────────────────────────────────
    cot_pred_bridges_all = None
    if eval_dataset_responses_path is not None:
        cot_records = []
        with jsonlines.open(eval_dataset_responses_path, 'r') as reader:
            for record in reader:
                cot_records.append(record)

        assert len(cot_records) == n_eval, (
            f"jsonl 文件样本数({len(cot_records)})与 dataset 样本数({n_eval})不一致"
        )

        eval_bridges_list = dataset[eval_bridge_col].astype(str).tolist()
        for i, (record, bridge_gt) in enumerate(zip(cot_records, eval_bridges_list)):
            assert record['correct_bridge'] == bridge_gt, (
                f"第{i}条样本 correct_bridge 不一致: "
                f"jsonl='{record['correct_bridge']}', dataset='{bridge_gt}'"
            )

        cot_pred_bridges_all = [record['pred_bridge'] for record in cot_records]
        print(f"成功读取 {len(cot_pred_bridges_all)} 条 CoT pred_bridge")

        # Pull CoT first-step text for dual-position probing (§C.1 pos 2).
        cot_first_steps_all = [
            _extract_first_cot_step(r.get("full_generation", ""))
            for r in cot_records
        ]
        have_first_step = any(s for s in cot_first_steps_all)
        if probe_cot_first_step and not have_first_step:
            print("WARN: probe_cot_first_step=True but no full_generation "
                  "text found in jsonl; disabling dual-position probing.")
            probe_cot_first_step = False

    eval_prompts  = dataset[eval_prompt_col].astype(str).tolist()
    eval_bridges  = dataset[eval_bridge_col].astype(str).tolist()
    eval_subjects = dataset[eval_subject_col].astype(str).tolist()

    # ── 预计算：correct bridge token ids ─────────────────────────────────
    use_idx = 0 if 'qwen3' in model.config.model_type.lower() else 1
    _fallback_tid = _safe_fallback_tid(tokenizer)

    correct_token_ids = []
    for bridge in eval_bridges:
        if not bridge or bridge in ('nan', 'null'):
            correct_token_ids.append(_fallback_tid)
        else:
            tids = tokenizer.encode(bridge.strip(), add_special_tokens=False)
            correct_token_ids.append(tids[0] if tids else _fallback_tid)

    # ── 预计算：CoT bridge token ids ─────────────────────────────────────
    cot_token_ids_all = None
    if cot_pred_bridges_all is not None:
        cot_token_ids_all = []
        for bridge in cot_pred_bridges_all:
            if not bridge or bridge in ('nan', 'null'):
                cot_token_ids_all.append(_fallback_tid)
            else:
                tokens = tokenizer.encode(str(bridge).strip(), add_special_tokens=False)
                cot_token_ids_all.append(tokens[0] if tokens else _fallback_tid)

    # ── 预计算：所有 batch 的 tokenization 和 subject position ───────────
    print("Pre-tokenizing all batches on CPU...")
    n_layers = model.config.num_hidden_layers
    # Use the single Table-7 best layer (paper §C.1 applies logit lens at
    # the same layer selected for the linear probe).
    best_layer = get_best_layer(model_name, dataset_name, n_layers=n_layers)
    source_layer_idxs = [best_layer]
    print(f"Logit lens: using Table-7 best layer {best_layer}/{n_layers-1} "
          f"for ({model_name}, {dataset_name})")
    E = model.get_output_embeddings().weight.detach()  # 留在 GPU

    batches = []
    for start_idx in tqdm(range(0, n_eval, batch_size), desc="Pre-tokenizing"):
        end_idx = min(start_idx + batch_size, n_eval)

        batch_prompts      = eval_prompts[start_idx:end_idx]
        batch_subjects     = eval_subjects[start_idx:end_idx]
        batch_correct_tids = correct_token_ids[start_idx:end_idx]
        batch_cot_tids     = (cot_token_ids_all[start_idx:end_idx]
                              if cot_token_ids_all is not None else None)

        # tokenization 在 CPU 上做，不 .to(device)
        prompt_inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        )

        last_subject_token_positions = (
            tokenization_utils.find_exact_substrings_token_positions_from_tensor(
                tokenizer, prompt_inputs["input_ids"], batch_subjects
            )
        )

        # ── Position (2): last subject token INSIDE the CoT first step ──
        cot_inputs = None
        cot_positions = None
        if probe_cot_first_step and cot_records is not None:
            batch_first_steps = cot_first_steps_all[start_idx:end_idx]
            # Build prompt + first_step; only for samples with non-empty step.
            batch_dual_prompts = [
                (p + "\n" + s) if s else p
                for p, s in zip(batch_prompts, batch_first_steps)
            ]
            cot_inputs = tokenizer(
                batch_dual_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=768
            )
            # only_last=True → returns the last occurrence of subject, i.e.
            # the subject mention inside the model's first reasoning step.
            cot_positions = (
                tokenization_utils.find_exact_substrings_token_positions_from_tensor(
                    tokenizer, cot_inputs["input_ids"], batch_subjects
                )
            )

        batches.append({
            'prompt_inputs': prompt_inputs,
            'positions':     last_subject_token_positions,
            'correct_tids':  batch_correct_tids,
            'cot_tids':      batch_cot_tids,
            'cot_inputs':    cot_inputs,
            'cot_positions': cot_positions,
            'start_idx':     start_idx,
            'end_idx':       end_idx,
        })

    print(f"Pre-tokenization done. {len(batches)} batches ready.")

    # ── 按 batch 收集结果 ─────────────────────────────────────────────────
    all_in_topk  = []
    all_ranks    = []
    all_top1     = []
    all_cot_tids = [] if cot_token_ids_all is not None else None

    for batch in tqdm(batches, desc="Logit lens batches"):
        prompt_inputs = {k: v.to(model.device) for k, v in batch['prompt_inputs'].items()}
        positions     = batch['positions']
        correct_tids  = batch['correct_tids']
        cot_tids      = batch['cot_tids']

        hidden_states = get_hidden_states(model, prompt_inputs)

        t1_hidden_states = torch.stack(
            [hidden_states[:, i, pos, :]
             for i, pos in enumerate(positions)],
            dim=1,
        )

        results_list = []
        for layer_i, t1_hs in enumerate(t1_hidden_states):
            if layer_i not in source_layer_idxs:
                continue
            values_map = t1_hs.matmul(E.T)
            layer_result = check_topk(values_map, correct_tids, k=top_k)
            results_list.append(layer_result)

        in_topk_p1, ranks_p1, top1_p1 = merge_results(results_list)

        # ── Position (2): CoT first-step subject-last-token ────────────
        if batch.get('cot_inputs') is not None:
            cot_inputs = {k: v.to(model.device)
                          for k, v in batch['cot_inputs'].items()}
            cot_positions = batch['cot_positions']
            cot_hidden = get_hidden_states(model, cot_inputs)
            t2_hidden_states = torch.stack(
                [cot_hidden[:, i, pos, :]
                 for i, pos in enumerate(cot_positions)],
                dim=1,
            )
            p2_list = []
            for layer_i, t2_hs in enumerate(t2_hidden_states):
                if layer_i not in source_layer_idxs:
                    continue
                values_map = t2_hs.matmul(E.T)
                p2_list.append(check_topk(values_map, correct_tids, k=top_k))
            in_topk_p2, ranks_p2, top1_p2 = merge_results(p2_list)

            # Union across positions (§C.1): in_topk OR; rank = min; top1
            # from the position with smaller rank.
            in_topk, ranks, top1 = [], [], []
            for a, b, ra, rb, t1a, t1b in zip(
                    in_topk_p1, in_topk_p2, ranks_p1, ranks_p2,
                    top1_p1, top1_p2):
                in_topk.append(a or b)
                if ra <= rb:
                    ranks.append(ra); top1.append(t1a)
                else:
                    ranks.append(rb); top1.append(t1b)
        else:
            in_topk, ranks, top1 = in_topk_p1, ranks_p1, top1_p1

        all_in_topk.extend(in_topk)
        all_ranks.extend(ranks)
        all_top1.extend(top1)

        if cot_tids is not None:
            all_cot_tids.extend(cot_tids)

    # ── 汇总基础指标 ──────────────────────────────────────────────────────
    recall_at_k = sum(all_in_topk) / n_eval
    mean_rank   = float(np.mean(all_ranks))

    print(f"Recall@{top_k}: {recall_at_k:.4f}  |  Mean Rank: {mean_rank:.2f}")

    # ── CPF 四分类（仅当提供 CoT 时）──────────────────────────────────────
    cpf = None
    all_cot_ranks = []
    if all_cot_tids:
        category_counts = {
            'faithful_correct':           0,  # 内部正确，CoT 正确
            'faithful_incorrect':         0,  # 内部错误，CoT 错误（两者一致）
            'internal_correct_cot_wrong': 0,  # 内部正确，CoT 撒谎
            'internal_wrong_cot_correct': 0,  # 内部错误，CoT lucky guess
        }

        rank_internal_prefers_correct = 0
        rank_internal_prefers_cot     = 0
        rank_comparison_n             = 0

        # ── 重跑前向，计算 CoT bridge 在各层的 rank（复用预计算好的 batches）─
        for batch in tqdm(batches, desc="Computing CoT ranks"):
            prompt_inputs  = {k: v.to(model.device) for k, v in batch['prompt_inputs'].items()}
            positions      = batch['positions']
            batch_cot_tids = batch['cot_tids']

            hidden_states = get_hidden_states(model, prompt_inputs)
            # shape: [n_layers, B, seq_len, hidden]，全程在 GPU 上

            # 从 seq_len 维取出每个样本 last subject token 位置的 hidden state
            t1_hidden_states = torch.stack(
                [hidden_states[:, i, pos, :]          # [n_layers, hidden]
                 for i, pos in enumerate(positions)],
                dim=1,
            )  # [n_layers, B, hidden]

            batch_cot_results = []
            for layer_i, t1_hs in enumerate(t1_hidden_states):
                if layer_i not in source_layer_idxs:
                    continue
                values_map = t1_hs.matmul(E.T)  # 保持在 GPU，check_topk 内部转 CPU
                _, layer_ranks, _ = check_topk(values_map, batch_cot_tids, k=top_k)
                batch_cot_results.append((
                    [False] * len(batch_cot_tids),
                    layer_ranks,
                    [None] * len(batch_cot_tids),
                ))

            _, min_cot_ranks, _ = merge_results(batch_cot_results)
            all_cot_ranks.extend(min_cot_ranks)

        # ── 四分类 + rank 比较 ────────────────────────────────────────────
        for i in range(n_eval):
            ll_top1  = all_top1[i]
            correct  = correct_token_ids[i]
            cot_tid  = all_cot_tids[i]

            internal_correct = (ll_top1 == correct)
            cot_correct      = (cot_tid  == correct)

            if internal_correct and cot_correct:
                category_counts['faithful_correct'] += 1
            elif internal_correct and not cot_correct:
                category_counts['internal_correct_cot_wrong'] += 1
            elif not internal_correct and cot_correct:
                category_counts['internal_wrong_cot_correct'] += 1
            else:
                category_counts['faithful_incorrect'] += 1

            # rank 比较（仅在 cot bridge ≠ correct bridge 的样本上统计）
            if cot_tid != correct:
                r_correct = all_ranks[i]
                r_cot     = all_cot_ranks[i]
                if r_correct < r_cot:
                    rank_internal_prefers_correct += 1
                elif r_cot < r_correct:
                    rank_internal_prefers_cot += 1
                rank_comparison_n += 1

        cot_recall_at_k = sum(r <= top_k for r in all_cot_ranks) / n_eval

        cpf = {
            'category_counts': category_counts,
            'category_rates':  {k: v / n_eval for k, v in category_counts.items()},
            f'cot_recall@{top_k}': cot_recall_at_k,
            'cot_mean_rank':   float(np.mean(all_cot_ranks)),
            'rank_comparison': {
                'n_samples_where_cot_ne_correct': rank_comparison_n,
                'rate_internal_prefers_correct':  (rank_internal_prefers_correct / rank_comparison_n
                                                   if rank_comparison_n > 0 else None),
                'rate_internal_prefers_cot':      (rank_internal_prefers_cot / rank_comparison_n
                                                   if rank_comparison_n > 0 else None),
            }
        }
        print(f"CPF Results: {cpf}")

    # ── 组装 metric ───────────────────────────────────────────────────────
    metric = {
        'model_name':       model_name,
        'dataset_name':     dataset_name,
        'interp_tool':      interp_tool,
        'seed':             seed,
        'top_k':            top_k,
        f'recall@{top_k}':  recall_at_k,
        'mean_rank':        mean_rank,
        'n_samples':        n_eval,
        'cpf':              cpf if cpf is not None else {},
        'per_sample': {
            'in_topk': all_in_topk,
            'ranks':   all_ranks,
            'top1':    all_top1,
        }
    }

    # ── 写回 dataset 列（方便后续分析）───────────────────────────────────
    dataset["logit_lens_in_topk"] = all_in_topk
    dataset["logit_lens_rank"]    = all_ranks

    # ── 按样本打标：是否内部表征真的用到了 CoT bridge ────────────────────
    if all_cot_tids and all_cot_ranks:
        labeled_records = []
        for i in range(n_eval):
            cot_rank   = all_cot_ranks[i]
            cot_tid    = all_cot_tids[i]
            correct_id = correct_token_ids[i]

            # 核心 label：CoT bridge 的 first token 排进了 top_k
            used_cot_bridge = (cot_rank <= top_k)  # 目前是用的这个，后面看看要不要修改

            labeled_records.append({
                "index":            i,
                "prompt":           eval_prompts[i],
                "correct_bridge":   eval_bridges[i],
                "pred_bridge":      cot_pred_bridges_all[i],
                "correct_token_id": correct_id,
                "cot_token_id":     cot_tid,
                "ll_top1_token_id": all_top1[i],
                "ll_rank_correct":  all_ranks[i],
                "ll_rank_cot":      cot_rank,
                "correct_in_topk":  all_in_topk[i],
                "used_cot_bridge":  used_cot_bridge,  # ← 核心 label
            })

        labeled_path = (
            f"{os.environ['SCRATCH']}/results/open-r1/twohop_results/"
            f"{dataset_name}_{model_name}_{interp_tool}_{seed}_results_labeled.jsonl"
        )
        os.makedirs(os.path.dirname(labeled_path), exist_ok=True)
        with jsonlines.open(labeled_path, mode="w") as writer:
            writer.write_all(labeled_records)
        print(f"Per-sample labels saved to {labeled_path}")

    # ── 保存结果到 jsonl ──────────────────────────────────────────────────
    output_dir = f"{os.environ['SCRATCH']}/results/open-r1/{dataset_name}_cpf_results"
    os.makedirs(output_dir, exist_ok=True)

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write(metric)

    print(f"Results saved to {output_path}")

    return metric

    # Other Results on All tokens after Subject token