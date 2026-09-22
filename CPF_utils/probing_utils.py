from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm  # 可选，用于进度条
import pandas as pd
from CPF_utils.patchscopes_utils import get_hidden_states
from CPF_utils.data_utils import load_dataset
from CPF_utils import tokenization_utils
from CPF_utils.layer_config import get_best_layer, get_probe_hparams
from CPF_utils.logitlens_utils import _extract_first_cot_step
import jsonlines


# 全局路径
SAVE_DIR = f"{os.environ['SCRATCH']}/results/open-r1/probing_results"
os.makedirs(SAVE_DIR, exist_ok=True)  # 自动创建目录

# 全局存储（从磁盘加载）
trained_probes = {}
probe_cache_file = os.path.join(SAVE_DIR, "trained_probes_cache.pth")

# 在函数开头加载缓存（如果存在）
if os.path.exists(probe_cache_file):
    print(f"Loading trained probes from {probe_cache_file}...")
    trained_probes = torch.load(probe_cache_file, map_location='cpu', weights_only=False)


# 预存的best layer dict（key改为 (model_name, train_dataset_name)）
best_layer_dict = {
    # 示例：{('Llama-2-7b-hf', 'onehop_dataset'): 21, ...}
    # 如果没有，会默认用中间层
}


def run_two_hop_linear_probe_evaluation(model, train_dataset, eval_dataset, train_dataset_name,
                                        eval_dataset_name, eval_dataset_responses_path,
                                        tokenizer, batch_size=64, top_k=200,
                                        output_path=None, seed=8888,
                                        probe_cot_first_step=True):
    """Linear probe eval for TwoHopFact.

    probe_cot_first_step: if True (default, paper §C.1 faithful), evaluates the
        probe at BOTH Position 1 (last token of e1 in subject_cut prompt) AND
        Position 2 (last token of e1 INSIDE the model's CoT first step), then
        UNION's top-K. Paired with logit lens 2-pos UNION methodology.
        Set False to revert to legacy 1-position (Pos 1 only) behavior.
        Requires eval_dataset_responses_path (CoT records) to compute Pos 2.
    """
    """
    - probe缓存key基于 (model_name, train_dataset_name)
    - metrics存为dict: {'train': train_metric, eval_name1: eval_metric1, ...}
    - 如果probe已训好，直接用；如果有对应eval_name的metric，直接返回；否则计算missing的eval metric
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # `model_name` here is used both as the (model, dataset) cache lookup key
    # and as the suffix in saved filenames (labeled jsonl, etc). For evaluating
    # a GRPO-trained checkpoint with the BASE model's cached probe weights we
    # need cache to point at the base name but filenames to use the trained
    # run's name. Split into two:
    output_model_name = getattr(model.config, '_name_or_path', 'unknown_model')
    if '/' in output_model_name:
        output_model_name = output_model_name.split('/')[-1]
    # PROBE_BASE_MODEL env var overrides only the cache lookup; if not set,
    # cache key == filename suffix (original behaviour).
    model_name = os.environ.get('PROBE_BASE_MODEL', output_model_name)
    if model_name != output_model_name:
        print(f"[override] PROBE_BASE_MODEL={model_name} (cache key); "
              f"filenames use {output_model_name}")

    interp_tool = "linear_probe"   # 供文件名使用，与 logit lens 对齐

    # 新key：基于train_dataset_name
    key = (model_name, train_dataset_name)

    # 关键修复：始终定义n_train和n_eval（无论cached与否）
    n_train = len(train_dataset)
    n_eval = len(eval_dataset)

    # 决定层（基于 Table 7）
    n_layers = len(model.model.layers)
    best_layer = best_layer_dict.get(key)
    if best_layer is None:
        # Fall back to Table 7 config; use train_dataset_name to pick task.
        best_layer = get_best_layer(model_name, train_dataset_name, n_layers=n_layers)
        print(f"Using Table 7 best layer {best_layer}/{n_layers-1} for ({model_name}, {train_dataset_name})")
    else:
        print(f"Using cached best layer {best_layer}/{n_layers-1}")

    # Pull probe hyperparameters from Table 7 (falls back to defaults).
    probe_hp = get_probe_hparams(model_name, train_dataset_name)

    # 检查train_dataset列
    train_prompt_col = 'r1(e1).prompt'
    train_bridge_col = 'e2.value'
    train_subject_col = 'r1(e1).subject_cut.prompt'
    for col in [train_prompt_col, train_bridge_col, train_subject_col]:
        if col not in train_dataset.columns:
            raise ValueError(f"train_dataset缺少必要列: {col}")

    # 检查eval_dataset列
    eval_prompt_col = 'r2(r1(e1)).prompt'
    eval_bridge_col = 'e2.value'
    eval_subject_col = 'r2(r1(e1)).subject_cut.prompt'
    for col in [eval_prompt_col, eval_bridge_col, eval_subject_col]:
        if col not in eval_dataset.columns:
            raise ValueError(f"eval_dataset缺少必要列: {col}")

    # ================== 如果probe已缓存 ==================
    if key in trained_probes:
        data = trained_probes[key]
        probe = data['probe'].to(device)
        cached_metrics = data.get('metrics', {})
        layer = data['layer']

        print(f"Linear Probe already trained for Model {model_name} with Train Dataset {train_dataset_name} (layer {layer})")

        # 如果已有当前eval_dataset_name的metric，直接返回完整metric.
        # SKIP_PROBE_CACHE_HIT lets callers (e.g. evaluating GRPO-trained
        # checkpoints with base-model probe weights) force a fresh per-sample
        # eval pass so a labeled jsonl is written for THIS run.
        if eval_dataset_name in cached_metrics and not os.environ.get('SKIP_PROBE_CACHE_HIT'):
            print(f"Metric for eval_dataset {eval_dataset_name} already cached, returning...")
            full_metric = cached_metrics.get('train', {}).copy()
            full_metric.update(cached_metrics[eval_dataset_name])
            full_metric['layer'] = layer
            return full_metric
        if os.environ.get('SKIP_PROBE_CACHE_HIT'):
            print('[override] SKIP_PROBE_CACHE_HIT — bypassing cache, will run per-sample eval.')

        # 否则：probe已训，只计算missing的eval metric（和train metric可选）
        print(f"Probe cached, but no metric for eval_dataset {eval_dataset_name}, computing eval metric...")

    else:
        # ================== 需要训练probe ==================
        print(f"starting to train Linear Probe for Model {model_name} using Train Dataset {train_dataset_name} for eval on {eval_dataset_name}...")

        # 训练集收集
        train_hs_list = []
        train_labels_chunk_list = []

        train_prompts = train_dataset[train_prompt_col].astype(str).tolist()
        train_bridges = train_dataset[train_bridge_col].astype(str).tolist()
        train_subjects = train_dataset[train_subject_col].astype(str).tolist()

        for start_idx in tqdm(range(0, n_train, batch_size), desc="Collecting train hidden states"):
            end_idx = min(start_idx + batch_size, n_train)
            batch_prompts = train_prompts[start_idx:end_idx]
            batch_bridges = train_bridges[start_idx:end_idx]
            batch_subjects = train_subjects[start_idx:end_idx]

            first_token_ids = []
            for bridge in batch_bridges:
                if not bridge or bridge in ('nan', 'null'):
                    first_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id)) #检测是否能在hs中看到first token of bridge
                    continue
                tokens = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                first_token_ids.append(tokens[0] if tokens else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
            train_labels_chunk_list.append(first_token_ids)

            inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

            last_subject_token_positions = tokenization_utils.find_exact_substrings_token_positions_from_tensor(
                tokenizer, inputs["input_ids"], batch_subjects
            )

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[best_layer + 1]

                seq_lens = inputs['attention_mask'].sum(dim=1) - 1
                positions = [seq_lens[i].item() if pos < 0 else pos for i, pos in enumerate(last_subject_token_positions)]
                positions = torch.tensor(positions, device=device)

                batch_indices = torch.arange(hidden.size(0), device=device)
                batch_hs = hidden[batch_indices, positions, :].float().cpu()
                train_hs_list.append(batch_hs)

        # 训练probe（分chunk）
        hidden_dim = train_hs_list[0].shape[1]
        vocab_size = len(tokenizer)

        probe = nn.Linear(hidden_dim, vocab_size).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(),
                                      lr=probe_hp["lr"],
                                      weight_decay=probe_hp["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        probe.train()
        n_epochs = probe_hp["epochs"]
        for epoch in range(n_epochs):
            total_loss = 0.0
            chunk_indices = np.random.permutation(len(train_hs_list))
            for i in chunk_indices:
                batch_hs = train_hs_list[i].to(device)
                batch_labels = torch.tensor(train_labels_chunk_list[i], device=device)

                optimizer.zero_grad()
                logits = probe(batch_hs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {total_loss / len(train_hs_list):.4f}")

        # 初始化cached data
        trained_probes[key] = {
            'probe': probe.cpu(),
            'layer': best_layer,
            'metrics': {}  # 将在此填充
        }

    def evaluate_on_dataset(hs_list, labels_chunk_list, dataset_name, n_samples,
                             pred_bridge_labels_chunk_list=None,
                             hs_list_p2=None):
        """Evaluate probe on collected hidden states.

        hs_list_p2: optional Position-2 hidden states (per paper §C.1). When
            provided, B_INT (correct_in_topk) and pred_in_topk become UNION
            across positions; ranks become min(rank_p1, rank_p2); probe_top1
            is taken from the position with smaller rank for correct bridge.
        """
        total_correct = 0
        pred_in_hs_total_correct = 0
        all_ranks = []
        pred_bridge_all_ranks = []
        use_2pos = hs_list_p2 is not None

        # faithfulness四分类计数
        category_counts = {
            'faithful_correct': 0,  # 内部正确，CoT正确
            'faithful_incorrect': 0,  # 内部错误，CoT错误（两者一致）
            'internal_correct_cot_wrong': 0,  # 内部正确，CoT撒谎了
            'internal_wrong_cot_correct': 0,  # 内部错误，CoT lucky guess
        }

        # rank比较计数（仅在cot bridge != correct bridge的样本上统计）
        rank_internal_prefers_correct = 0  # rank_correct < rank_cot
        rank_internal_prefers_cot = 0  # rank_cot < rank_correct
        rank_comparison_n = 0

        # 用于 per-sample 打标（仅在有 CoT 时收集）
        per_sample_top1       = []  # probe top-1 token id
        per_sample_cr         = []  # rank of correct bridge
        per_sample_cotr       = []  # rank of cot bridge
        per_sample_cot_intopk = []  # cot bridge in top-k?

        probe.to(device)
        probe.eval()
        with torch.no_grad():
            for i in tqdm(range(len(hs_list)), desc=f"Evaluating on {dataset_name}"):
                batch_hs = hs_list[i].to(device)
                batch_labels = torch.tensor(labels_chunk_list[i], device=device)  # [B]

                logits = probe(batch_hs)  # [B, vocab]
                sorted_indices = logits.argsort(dim=-1, descending=True)  # [B, vocab]

                # ── Recall@K（correct bridge） — Position 1 ────────────────────
                topk_indices_p1 = sorted_indices[:, :top_k]  # [B, top_k]
                correct_p1 = (topk_indices_p1 == batch_labels.unsqueeze(1)).any(dim=1)

                # ── Rank of correct bridge at Pos 1 ────────────────────────────
                correct_match_p1 = (sorted_indices == batch_labels.unsqueeze(1))
                correct_ranks_p1 = correct_match_p1.float().argmax(dim=1) + 1

                # ── Position 2 (paper §C.1 UNION) ──────────────────────────────
                if use_2pos:
                    batch_hs_p2 = hs_list_p2[i].to(device)
                    logits_p2 = probe(batch_hs_p2)
                    sorted_indices_p2 = logits_p2.argsort(dim=-1, descending=True)
                    topk_indices_p2 = sorted_indices_p2[:, :top_k]
                    correct_p2 = (topk_indices_p2 == batch_labels.unsqueeze(1)).any(dim=1)
                    correct_match_p2 = (sorted_indices_p2 == batch_labels.unsqueeze(1))
                    correct_ranks_p2 = correct_match_p2.float().argmax(dim=1) + 1

                    # UNION: B_INT=1 iff in top-K at EITHER position
                    correct = correct_p1 | correct_p2
                    # rank = min across positions (better rank wins)
                    correct_ranks = torch.minimum(correct_ranks_p1, correct_ranks_p2)
                    # probe_top1 = top1 of position with lower correct_rank
                    # (whichever position recognises bridge better)
                    use_p2_top1 = (correct_ranks_p2 < correct_ranks_p1)
                    probe_top1 = torch.where(use_p2_top1,
                                              sorted_indices_p2[:, 0],
                                              sorted_indices[:, 0])
                else:
                    correct = correct_p1
                    correct_ranks = correct_ranks_p1
                    probe_top1 = sorted_indices[:, 0]

                total_correct += correct.sum().item()
                all_ranks.extend(correct_ranks.cpu().tolist())

                # ── 以下仅在提供了cot pred_bridge时执行 ───────────────────────
                if pred_bridge_labels_chunk_list is not None:
                    pred_bridge_labels = torch.tensor(pred_bridge_labels_chunk_list[i], device=device)  # [B]

                    # Recall@K（cot bridge在probe top-k里的比例） — UNION 2-pos
                    pred_in_hs_p1 = (topk_indices_p1 == pred_bridge_labels.unsqueeze(1)).any(dim=1)
                    cot_match_p1 = (sorted_indices == pred_bridge_labels.unsqueeze(1))
                    cot_ranks_p1 = cot_match_p1.float().argmax(dim=1) + 1
                    if use_2pos:
                        pred_in_hs_p2 = (topk_indices_p2 == pred_bridge_labels.unsqueeze(1)).any(dim=1)
                        cot_match_p2 = (sorted_indices_p2 == pred_bridge_labels.unsqueeze(1))
                        cot_ranks_p2 = cot_match_p2.float().argmax(dim=1) + 1
                        pred_in_hs = pred_in_hs_p1 | pred_in_hs_p2
                        cot_ranks = torch.minimum(cot_ranks_p1, cot_ranks_p2)
                    else:
                        pred_in_hs = pred_in_hs_p1
                        cot_ranks = cot_ranks_p1
                    pred_in_hs_total_correct += pred_in_hs.sum().item()
                    pred_bridge_all_ranks.extend(cot_ranks.cpu().tolist())

                    # ── Faithfulness四分类 ─────────────────────────────────────
                    # 用probe top-1作为"内部bridge"的代理 (already 2-pos aware above)
                    internal_correct = (probe_top1 == batch_labels)  # [B] bool
                    cot_correct = (pred_bridge_labels == batch_labels)  # [B] bool

                    for b in range(batch_labels.size(0)):
                        ic, cc = internal_correct[b].item(), cot_correct[b].item()
                        if ic and cc:
                            category_counts['faithful_correct'] += 1
                        elif ic and not cc:
                            category_counts['internal_correct_cot_wrong'] += 1
                        elif not ic and cc:
                            category_counts['internal_wrong_cot_correct'] += 1
                        else:
                            category_counts['faithful_incorrect'] += 1

                    # ── Rank比较（仅在cot bridge ≠ correct bridge的样本）────────
                    different_mask = (pred_bridge_labels != batch_labels)  # [B]
                    if different_mask.any():
                        diff_correct_ranks = correct_ranks[different_mask]
                        diff_cot_ranks = cot_ranks[different_mask]
                        rank_internal_prefers_correct += (diff_correct_ranks < diff_cot_ranks).sum().item()
                        rank_internal_prefers_cot += (diff_cot_ranks < diff_correct_ranks).sum().item()
                        rank_comparison_n += different_mask.sum().item()

                    # ── 收集 per-sample 打标所需数据 ──────────────────────────
                    per_sample_top1.extend(probe_top1.cpu().tolist())
                    per_sample_cr.extend(correct_ranks.cpu().tolist())
                    per_sample_cotr.extend(cot_ranks.cpu().tolist())
                    per_sample_cot_intopk.extend(pred_in_hs.cpu().tolist())

        # ── 汇总指标 ──────────────────────────────────────────────────────────
        recall_at_k = total_correct / n_samples
        mean_rank = np.mean(all_ranks)

        if pred_bridge_labels_chunk_list is None:
            return recall_at_k, mean_rank

        cpf = {
            'category_counts': category_counts,
            'category_rates': {k: v / n_samples for k, v in category_counts.items()},
            f'pred_recall@{top_k}': pred_in_hs_total_correct / n_samples,
            'pred_mean_rank': np.mean(pred_bridge_all_ranks),
            'rank_comparison': {
                'n_samples_where_cot_ne_correct': rank_comparison_n,
                'rate_internal_prefers_correct': rank_internal_prefers_correct / rank_comparison_n if rank_comparison_n > 0 else None,
                'rate_internal_prefers_cot': rank_internal_prefers_cot / rank_comparison_n if rank_comparison_n > 0 else None,
            }
        }

        per_sample_info = {
            'probe_top1':  per_sample_top1,
            'rank_correct': per_sample_cr,
            'rank_cot':    per_sample_cotr,
            'cot_in_topk': per_sample_cot_intopk,
        }

        return recall_at_k, mean_rank, cpf, per_sample_info

    # ================== 计算train metric（始终计算或缓存） ==================
    if 'train' not in trained_probes[key]['metrics']:
        print("Computing train metric...")
        # 如果cached但无train metric，重新收集train hs（train_hs_list可能不存在，所以重新收集）
        train_hs_list = []
        train_labels_chunk_list = []

        train_prompts = train_dataset[train_prompt_col].astype(str).tolist()
        train_bridges = train_dataset[train_bridge_col].astype(str).tolist()
        train_subjects = train_dataset[train_subject_col].astype(str).tolist()

        for start_idx in tqdm(range(0, n_train, batch_size), desc="Re-collecting train hidden states for metric"):
            end_idx = min(start_idx + batch_size, n_train)
            batch_prompts = train_prompts[start_idx:end_idx]
            batch_bridges = train_bridges[start_idx:end_idx]
            batch_subjects = train_subjects[start_idx:end_idx]

            first_token_ids = []
            for bridge in batch_bridges:
                if not bridge or bridge in ('nan', 'null'):
                    first_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
                    continue
                tokens = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                first_token_ids.append(tokens[0] if tokens else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
            train_labels_chunk_list.append(first_token_ids)

            inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

            last_subject_token_positions = tokenization_utils.find_exact_substrings_token_positions_from_tensor(
                tokenizer, inputs["input_ids"], batch_subjects
            )

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[best_layer + 1]

                seq_lens = inputs['attention_mask'].sum(dim=1) - 1
                positions = [seq_lens[i].item() if pos < 0 else pos for i, pos in enumerate(last_subject_token_positions)]
                positions = torch.tensor(positions, device=device)

                batch_indices = torch.arange(hidden.size(0), device=device)
                batch_hs = hidden[batch_indices, positions, :].float().cpu()
                train_hs_list.append(batch_hs)

        train_recall_at_k, train_mean_rank = evaluate_on_dataset(
            train_hs_list, train_labels_chunk_list, "train_dataset", n_train
        )
        trained_probes[key]['metrics']['train'] = {
            'train_recall@{}'.format(top_k): train_recall_at_k,
            'train_mean_rank': train_mean_rank,
            'train_n_samples': n_train
        }
    else:
        train_metric = trained_probes[key]['metrics']['train']
        train_recall_at_k = train_metric['train_recall@{}'.format(top_k)]
        train_mean_rank = train_metric['train_mean_rank']

    # ================== 计算eval metric（如果missing） ==================
    cpf             = None
    per_sample_info = None
    cot_pred_bridges = None

    # SKIP_PROBE_CACHE_HIT also bypasses this second gate so the per-sample
    # eval runs fresh on the current model and writes a labeled jsonl.
    if eval_dataset_name not in trained_probes[key]['metrics'] or os.environ.get('SKIP_PROBE_CACHE_HIT'):
        print(f"Computing metric for eval_dataset {eval_dataset_name}...")
        eval_hs_list = []
        eval_labels_chunk_list = []
        pred_bridge_labels_chunk_list = []

        eval_prompts = eval_dataset[eval_prompt_col].astype(str).tolist()
        eval_bridges = eval_dataset[eval_bridge_col].astype(str).tolist()
        eval_subjects = eval_dataset[eval_subject_col].astype(str).tolist()
        print(f"The size of eval_dataset {eval_dataset_name} is {n_eval} samples.")

        if eval_dataset_responses_path is not None:
            # 读取jsonl文件
            cot_records = []
            with jsonlines.open(eval_dataset_responses_path, 'r') as reader:
                for record in reader:
                    cot_records.append(record)

            assert len(cot_records) == n_eval, (
                f"jsonl文件样本数({len(cot_records)})与eval_dataset样本数({n_eval})不一致"
            )

            # 验证correct_bridge与eval_dataset的e2.value次序一致
            for i, (record, bridge_gt) in enumerate(zip(cot_records, eval_bridges)):
                assert record['correct_bridge'] == bridge_gt, (
                    f"第{i}条样本的correct_bridge不一致: "
                    f"jsonl中为'{record['correct_bridge']}', "
                    f"eval_dataset中为'{bridge_gt}'"
                )

            # 提取pred_bridge列表，留着循环里按index使用
            cot_pred_bridges = [record['pred_bridge'] for record in cot_records]
            print(f"成功读取{len(cot_pred_bridges)}条CoT pred_bridge")

            # ── Position 2 prep (paper §C.1 2-pos UNION) ────────────────────
            # Extract first CoT step per sample for Pos 2 forward pass.
            # If probe_cot_first_step=False, skip — preserves legacy 1-pos behaviour.
            cot_first_steps = None
            if probe_cot_first_step:
                cot_first_steps = [_extract_first_cot_step(r.get("full_generation", ""))
                                    for r in cot_records]
                have_any = any(s for s in cot_first_steps)
                if not have_any:
                    print("WARN: probe_cot_first_step=True but jsonl has no "
                          "full_generation; disabling Pos 2.")
                    cot_first_steps = None
                else:
                    print(f"Pos 2 enabled: will probe at last e1 token inside "
                          f"CoT first step for each of {len(cot_first_steps)} samples.")

        # Pos 2 hidden states (None if disabled / no CoT)
        eval_hs_list_p2 = [] if (probe_cot_first_step and cot_first_steps is not None) else None

        for start_idx in tqdm(range(0, n_eval, batch_size), desc="Collecting eval hidden states"):
            end_idx = min(start_idx + batch_size, n_eval)
            batch_prompts = eval_prompts[start_idx:end_idx]
            batch_bridges = eval_bridges[start_idx:end_idx]
            batch_subjects = eval_subjects[start_idx:end_idx]
            batch_cot_pred_bridges = cot_pred_bridges[start_idx:end_idx]

            first_token_ids = []
            for bridge in batch_bridges:
                if not bridge or bridge in ('nan', 'null'):
                    first_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
                    continue
                tokens = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                first_token_ids.append(tokens[0] if tokens else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
            eval_labels_chunk_list.append(first_token_ids)

            # storing first token ids of cot_pred_bridges
            cot_pred_bridges_first_token_ids = []
            for bridge in batch_cot_pred_bridges:
                if not bridge or bridge in ('nan', 'null'):
                    cot_pred_bridges_first_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
                    continue
                tokens = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                cot_pred_bridges_first_token_ids.append(tokens[0] if tokens else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
            pred_bridge_labels_chunk_list.append(cot_pred_bridges_first_token_ids)

            inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

            last_subject_token_positions = tokenization_utils.find_exact_substrings_token_positions_from_tensor(
                tokenizer, inputs["input_ids"], batch_subjects
            )

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[best_layer + 1]

                seq_lens = inputs['attention_mask'].sum(dim=1) - 1
                positions = [seq_lens[i].item() if pos < 0 else pos for i, pos in enumerate(last_subject_token_positions)]
                positions = torch.tensor(positions, device=device)

                batch_indices = torch.arange(hidden.size(0), device=device)
                batch_hs = hidden[batch_indices, positions, :].float().cpu()
                eval_hs_list.append(batch_hs)

            # ── Position 2: forward (prompt + "\n" + first_step), probe at
            # the LAST occurrence of subject inside the CoT step ──
            if eval_hs_list_p2 is not None:
                batch_first_steps = cot_first_steps[start_idx:end_idx]
                batch_dual_prompts = [(p + "\n" + s) if s else p
                                       for p, s in zip(batch_prompts, batch_first_steps)]
                cot_inputs = tokenizer(batch_dual_prompts, padding=True, truncation=True,
                                        return_tensors='pt', max_length=768).to(device)
                cot_positions_raw = tokenization_utils.find_exact_substrings_token_positions_from_tensor(
                    tokenizer, cot_inputs["input_ids"], batch_subjects
                )
                with torch.no_grad():
                    outputs_p2 = model(**cot_inputs, output_hidden_states=True)
                    hidden_p2 = outputs_p2.hidden_states[best_layer + 1]
                    seq_lens_p2 = cot_inputs['attention_mask'].sum(dim=1) - 1
                    positions_p2 = [seq_lens_p2[i].item() if pos < 0 else pos
                                     for i, pos in enumerate(cot_positions_raw)]
                    positions_p2 = torch.tensor(positions_p2, device=device)
                    batch_indices_p2 = torch.arange(hidden_p2.size(0), device=device)
                    batch_hs_p2 = hidden_p2[batch_indices_p2, positions_p2, :].float().cpu()
                    eval_hs_list_p2.append(batch_hs_p2)

        if eval_dataset_responses_path is not None:
            eval_recall_at_k, eval_mean_rank, cpf, per_sample_info = evaluate_on_dataset(
                eval_hs_list, eval_labels_chunk_list, "eval_dataset", n_eval,
                pred_bridge_labels_chunk_list, hs_list_p2=eval_hs_list_p2
            )
        else:
            eval_recall_at_k, eval_mean_rank = evaluate_on_dataset(
                eval_hs_list, eval_labels_chunk_list, "eval_dataset", n_eval
            )

        trained_probes[key]['metrics'][eval_dataset_name] = {
            'eval_recall@{}'.format(top_k): eval_recall_at_k,
            'eval_mean_rank': eval_mean_rank,
            'eval_n_samples': n_eval,
            'cpf': cpf if cpf is not None else {},
        }
    else:
        eval_metric = trained_probes[key]['metrics'][eval_dataset_name]
        eval_recall_at_k = eval_metric['eval_recall@{}'.format(top_k)]
        eval_mean_rank = eval_metric['eval_mean_rank']

    # ── per-sample 打标并写出 labeled jsonl ──────────────────────────────────
    if per_sample_info is not None and cot_pred_bridges is not None:
        # 计算 correct bridge 的 first token id
        correct_token_ids = []
        for bridge in eval_bridges:
            if not bridge or bridge in ('nan', 'null'):
                correct_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
            else:
                tids = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                correct_token_ids.append(tids[0] if tids else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))

        # 计算 cot pred bridge 的 first token id
        cot_token_ids = []
        for bridge in cot_pred_bridges:
            if not bridge or bridge in ('nan', 'null'):
                cot_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
            else:
                tids = tokenizer.encode(bridge.strip(), add_special_tokens=False)
                cot_token_ids.append(tids[0] if tids else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))

        labeled_records = []
        for i in range(n_eval):
            probe_top1_tid = per_sample_info['probe_top1'][i]
            cot_tid        = cot_token_ids[i]
            correct_tid    = correct_token_ids[i]

            labeled_records.append({
                "index":                i,
                "prompt":               eval_prompts[i],
                "correct_bridge":       eval_bridges[i],
                "pred_bridge":          cot_pred_bridges[i],
                "correct_token_id":     correct_tid,
                "cot_token_id":         cot_tid,
                "probe_top1_token_id":  probe_top1_tid,        # 内部预测的 bridge
                "ll_rank_correct":      per_sample_info['rank_correct'][i],
                "ll_rank_cot":          per_sample_info['rank_cot'][i],
                "correct_in_topk":      bool(per_sample_info['rank_correct'][i] <= top_k),
                "used_cot_bridge":      bool(per_sample_info['cot_in_topk'][i]),   # CoT bridge 进入 probe top-k
                # ── CPF 核心 label：内部表征与 CoT 是否一致 ──
                "cot_matches_internal": (probe_top1_tid == cot_tid),   # probe top-1 == CoT pred（faithful）
                "internal_correct":     (probe_top1_tid == correct_tid),
                "cot_correct":          (cot_tid == correct_tid),
            })

        labeled_path = (
            f"{os.environ['SCRATCH']}/results/open-r1/twohop_results/"
            f"{eval_dataset_name}_{output_model_name}_{seed}_results_{interp_tool}_labeled.jsonl"
        )
        os.makedirs(os.path.dirname(labeled_path), exist_ok=True)
        with jsonlines.open(labeled_path, mode="w") as writer:
            writer.write_all(labeled_records)
        print(f"Per-sample labels saved to {labeled_path}")

    # ================== 最终metric（合并train + 当前eval） ==================
    metric = {
        'layer': best_layer,
        'train_recall@{}'.format(top_k): train_recall_at_k,
        'train_mean_rank': train_mean_rank,
        'train_n_samples': n_train,
        'eval_recall@{}'.format(top_k): eval_recall_at_k,
        'eval_mean_rank': eval_mean_rank,
        'eval_n_samples': n_eval,
        'cpf': cpf if cpf is not None else {},
    }
    print(f"Evaluation finished! "
          f"Train Recall@{top_k}: {train_recall_at_k:.4f}, Train Mean Rank: {train_mean_rank:.2f} "
          f"Eval ({eval_dataset_name}) Recall@{top_k}: {eval_recall_at_k:.4f}, Eval Mean Rank: {eval_mean_rank:.2f}")
    print(f"CPF Results: {cpf if cpf is not None else 'None'}")

    # 保存更新后的缓存 — re-load disk first to merge in any other tasks'
    # writes (defense against parallel array runs hitting same file).
    # SKIP_PROBE_CACHE_HIT mode = evaluating a different model with cached
    # base-model probe; do NOT save back (would corrupt base entry's metrics).
    if os.environ.get('SKIP_PROBE_CACHE_HIT'):
        print('[override] SKIP_PROBE_CACHE_HIT — skipping cache write to avoid corrupting base entry.')
    else:
        print('Saving trained probes to disk...')
        if os.path.exists(probe_cache_file):
            try:
                disk_probes = torch.load(probe_cache_file, map_location='cpu', weights_only=False)
                for k, v in disk_probes.items():
                    if k not in trained_probes:
                        trained_probes[k] = v
            except Exception as e:
                print(f"WARN: could not merge existing cache ({e}); overwriting.")
        torch.save(trained_probes, probe_cache_file)
        print(f"Trained probes updated and saved to {probe_cache_file}")

    # ── 保存结果到 jsonl ──────────────────────────────────────────────────
    output_dir = f"{os.environ['SCRATCH']}/results/open-r1/{eval_dataset_name}_{model_name}_{interp_tool}_{seed}_cpf_results"
    os.makedirs(output_dir, exist_ok=True)

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write(metric)

    print(f"Results saved to {output_path}")

    return metric




# Hint Intervention
# 这个很好判断hiddenstates内部，是否调用了Hint，来实现对应的任务


# ====================== Hint Intervention Probe ======================
def run_hint_linear_probe_evaluation(model, dataset: pd.DataFrame, dataset_name, tokenizer, batch_size=64):
    """
    Hint Intervention binary probe:
    - 判断hidden state是否“调用了hint”（即依赖hint改变回答）
    - 假设dataset (DataFrame) 包含以下列：
        - 'prompt_with_hint': 带hint的prompt
        - 'prompt_without_hint': 不带hint的prompt
        - 'answer_changed': bool/int列，1=带hint后回答改变（positive），0=未改变（negative）
    - 只收集answer_changed==1的with_hint hidden states作为positive
    - 收集answer_changed==0的with_hint hidden states + 所有without_hint作为negative
    - 训练binary linear probe (hidden_dim -> 2)
    - 返回metric: acc, positive_prob_mean等
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    model_name = getattr(model.config, '_name_or_path', 'unknown_model').split('/')[-1]
    key = (model_name, dataset_name + '_hint')

    if key in trained_probes:
        data = trained_probes[key]
        print(f"Hint Probe already trained for Model {model_name} on Dataset {dataset_name} (layer {data['layer']})")
        return data['metric']

    print(f"starting to train Hint Intervention Linear Probe for Model {model_name} on Dataset {dataset_name}...")

    # 检查必要列
    req_cols = ['prompt_with_hint', 'prompt_without_hint', 'answer_changed']
    for col in req_cols:
        if col not in dataset.columns:
            raise ValueError(f"DataFrame缺少必要列: {col}")

    # 决定层（同two-hop）
    n_layers = len(model.model.layers)
    best_layer = best_layer_dict.get(key, n_layers // 2)
    print(f"Using layer {best_layer}/{n_layers-1}")

    # 提取并构建positive/negative samples
    hs_pos = []  # with_hint 且 answer_changed==1
    hs_neg = []  # with_hint 且 answer_changed==0 + 所有 without_hint

    prompts_with = dataset['prompt_with_hint'].astype(str).tolist()
    prompts_without = dataset['prompt_without_hint'].astype(str).tolist()
    changed = dataset['answer_changed'].astype(int).tolist()
    n_samples = len(prompts_with)

    # 先收集所有without_hint hidden states（都作为negative）
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Collecting without_hint hidden states (negative)"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_prompts = prompts_without[start_idx:end_idx]

        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[best_layer + 1]
            seq_lens = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=device)
            batch_hs = hidden[batch_indices, seq_lens, :].float().cpu()
            hs_neg.append(batch_hs)

    # 再收集with_hint hidden states（根据changed分正负）
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Collecting with_hint hidden states"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_prompts = prompts_with[start_idx:end_idx]
        batch_changed = changed[start_idx:end_idx]

        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[best_layer + 1]
            seq_lens = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=device)
            batch_hs = hidden[batch_indices, seq_lens, :].float().cpu()

            # 分割正负
            for i, ch in enumerate(batch_changed):
                if ch == 1:
                    hs_pos.append(batch_hs[i:i+1])
                else:
                    hs_neg.append(batch_hs[i:i+1])

    # 合并
    if len(hs_pos) == 0:
        raise ValueError("No positive samples (answer_changed==1) found!")
    hs_pos = torch.cat(hs_pos, dim=0)
    hs_neg = torch.cat(hs_neg, dim=0)

    # 平衡采样（取min大小）
    min_size = min(len(hs_pos), len(hs_neg))
    hs_pos = hs_pos[:min_size]
    hs_neg = hs_neg[:min_size]

    hs = torch.cat([hs_pos, hs_neg], dim=0)
    labels = torch.cat([torch.ones(len(hs_pos)), torch.zeros(len(hs_neg))]).long()

    # ================== 训练Binary Probe ==================
    hidden_dim = hs.shape[1]
    probe = nn.Linear(hidden_dim, 2).to(device)  # binary
    hs_device = hs.to(device)

    dataset_probe = TensorDataset(hs_device, labels.to(device))
    train_loader = DataLoader(dataset_probe, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    n_epochs = 20
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_hs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_hs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {total_loss / len(train_loader):.4f}")

    # ================== 评估 ==================
    probe.eval()
    with torch.no_grad():
        logits = probe(hs_device)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels.to(device)).float().mean().item()
        pos_prob = torch.softmax(logits, dim=-1)[:, 1].mean().item()  # positive class概率均值

    metric = {
        'layer': best_layer,
        'accuracy': acc,
        'mean_positive_prob': pos_prob,
        'n_pos': len(hs_pos),
        'n_neg': len(hs_neg)
    }
    print(f"Hint Probe Training finished! Acc: {acc:.4f}, Mean Positive Prob: {pos_prob:.4f}")

    trained_probes[key] = {'probe': probe.cpu(), 'layer': best_layer, 'metric': metric}
    return metric


# ====================== Math/Multiplication Probe ======================
def run_math_linear_probe_evaluation(model, dataset: pd.DataFrame, dataset_name, tokenizer, batch_size=64, top_k=200):
    """
    Math/Multiplication probe（类似two-hop）:
    - 判断hidden state是否包含特定中间数字/计算步骤
    - 假设dataset (DataFrame) 包含列：
        - 'prompt': multiplication prompt（e.g., "123 * 456 = "）
        - 'intermediate': 中间计算步骤的数字文本（e.g., "56088" 或 partial product）
    - 预测intermediate的first token（同two-hop bridge）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    model_name = getattr(model.config, '_name_or_path', 'unknown_model').split('/')[-1]
    key = (model_name, dataset_name + '_math')

    if key in trained_probes:
        data = trained_probes[key]
        print(f"Math Probe already trained for Model {model_name} on Dataset {dataset_name} (layer {data['layer']})")
        return data['metric']

    print(f"starting to train Math Intermediate Linear Probe for Model {model_name} on Dataset {dataset_name}...")

    # 检查列
    prompt_col = 'prompt'
    intermediate_col = 'intermediate'
    if prompt_col not in dataset.columns or intermediate_col not in dataset.columns:
        raise ValueError(f"DataFrame缺少必要列: {prompt_col} 或 {intermediate_col}")

    # 决定层
    n_layers = len(model.model.layers)
    best_layer = best_layer_dict.get(key, n_layers // 2)
    print(f"Using layer {best_layer}/{n_layers-1}")

    prompts = dataset[prompt_col].astype(str).tolist()
    intermediates = dataset[intermediate_col].astype(str).tolist()
    n_samples = len(prompts)

    # ================== 收集hidden states ==================
    hs_list = []
    labels_list = []

    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Collecting hidden states"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_prompts = prompts[start_idx:end_idx]
        batch_inter = intermediates[start_idx:end_idx]

        first_token_ids = []
        for inter in batch_inter:
            if not inter or inter == 'nan':
                first_token_ids.append((tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
                continue
            tokens = tokenizer.encode(inter.strip(), add_special_tokens=False)
            first_token_ids.append(tokens[0] if tokens else (tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id))
        labels_list.extend(first_token_ids)

        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[best_layer + 1]
            seq_lens = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=device)
            batch_hs = hidden[batch_indices, seq_lens, :].float().cpu()
            hs_list.append(batch_hs)

    hs = torch.cat(hs_list, dim=0)
    labels = torch.tensor(labels_list)

    # ================== 训练 & 评估（同two-hop） ==================
    hidden_dim = hs.shape[1]
    vocab_size = len(tokenizer)

    probe = nn.Linear(hidden_dim, vocab_size).to(device)
    hs_device = hs.to(device)

    probe_dataset = TensorDataset(hs_device, labels.to(device))
    train_loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    n_epochs = 20
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_hs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_hs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {total_loss / len(train_loader):.4f}")

    probe.eval()
    with torch.no_grad():
        logits = probe(hs_device)
        topk_indices = torch.topk(logits, k=top_k, dim=-1).indices
        correct = (topk_indices == labels.to(device).unsqueeze(1)).any(dim=1)
        recall_at_k = correct.float().mean().item()

        ranks = [(logits[i].argsort(descending=True) == labels[i].to(device)).nonzero().item() + 1 for i in range(logits.shape[0])]
        mean_rank = np.mean(ranks)

    metric = {
        'layer': best_layer,
        'recall@{}'.format(top_k): recall_at_k,
        'mean_rank': mean_rank,
        'n_samples': n_samples
    }
    print(f"Math Probe Training finished! Recall@{top_k}: {recall_at_k:.4f}, Mean Rank: {mean_rank:.2f}")

    trained_probes[key] = {'probe': probe.cpu(), 'layer': best_layer, 'metric': metric}
    return metric



# tasks_data = {
#     "task1": {"X_train": X_train1, "y_train": y_train1, "X_test": X_test1, "y_test": y_test1, "max_iter": 500},
#     "task2": {"X_train": X_train2, "y_train": y_train2, "X_test": X_test2, "y_test": y_test2, "max_iter": 1000},
#     "task3": {"X_train": X_train3, "y_train": y_train3, "X_test": X_test3, "y_test": y_test3, "max_iter": 1500},
# }
#
# probe = LinearProbe()
#
# for task_name, data in tasks_data.items():
#     probe.fit(data["X_train"], data["y_train"], task_name, max_iter=data["max_iter"])
#     result = probe.evaluate(data["X_test"], data["y_test"], task_name)
#     print(f"{task_name} accuracy: {result['linear_probe_acc']}, training info: {result['task_info']}")