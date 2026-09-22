# CIA 最小复现代码：RS-B、DPO-A 与评测

[English README](README.md)

这个目录包含三个任务（TwoHop、Hint、Mult）上两种离线训练方法（RS-B、DPO-A）和对应评测的全部代码。它是从完整项目仓库里按实际调用关系抽出来的最小子集，只含代码，不含数据、模型权重、probe 权重和任何实验结果。

## 1. 要测的是什么

每条模型回答有两个二值标签：

- **B_INT**：模型内部是否真的用到了关键信息，由线性 probe 或因果干预判断。
- **B_CoT**：思维链（CoT）里有没有把这件事说出来。

**CIA** 是 B_INT 与 B_CoT 一致性的 macro-F1。faith = 1[B_INT == B_CoT]，acc 表示答案是否正确。训练时给 rollout 打标签，和评测用的是同一套定义。

| 任务 | B_INT | B_CoT |
|---|---|---|
| TwoHop | 线性 probe 在最后一个 e1 token 上读出 bridge entity（K = 100） | CoT 里是否说出 bridge entity |
| Hint | probe 在答案字母位置读出是否用了 hint | LLM judge（Qwen2.5-32B-Instruct）判断 CoT 是否承认用了 hint |
| Mult | 训练标签：probe 读求和前的 partial product。评测主指标：因果 corruption，改掉 partial product 后最终答案是否跟着变 | 最终答案是否等于 pp1 + pp2（自洽检查） |

## 2. 两种训练方法

两种方法共用同一批数据：base 模型对每个训练 prompt 采样 16 条（T = 1.0，top-p 1.0），每条按上表打标签。

| 方法 | 训练数据 | 训练方式 | 超参 |
|---|---|---|---|
| **RS-B** | 只保留 faith = 1 的回答，不要求答对 | SFT（`open-r1/src/open_r1/sft.py`） | lr 1e-6，cosine，warmup 0.1，1 或 2 epoch（见各 recipe） |
| **DPO-A** | 每个 prompt 按 acc + faith 排序，取前 3 条对后 3 条 | DPO（`open-r1/src/open_r1/dpo.py`） | β 0.1，lr 5e-7，1 epoch，sigmoid loss |

- TwoHop 的 recipe 每 15 步存一个 checkpoint，评测时在 VAL 上选 peak。Hint 和 Mult 只保存最终模型。
- 每个任务 × 模型的 recipe 在 `open-r1/recipes/CIA/rs/`（RS-B）和 `open-r1/recipes/CIA/dpo/`（DPO-A）。
- 训练用 accelerate + DeepSpeed ZeRO-3，配置在 `open-r1/recipes/accelerate_configs/zero3_no_offload_tuned.yaml`。

## 3. 目录结构

```
run/                     入口脚本（从这里开始）
  env.sh                 共用环境变量
  train_offline.sh       rollout → 打标签 → 构建 RS-B / DPO-A 数据 → 训练
  eval_twohop.sh         TwoHop 评测
  eval_hint.sh           Hint 评测
  eval_mult.sh           Mult 评测第 1 步：生成 + probe 版 CIA（次要指标）
  eval_mult_causal.sh    Mult 评测第 2 步：因果 corruption 指标（主指标）
  paired_bootstrap.sh    训练模型 vs base 的 paired bootstrap
open-r1/                 训练框架（fork 自 HuggingFace open-r1，Apache-2.0，见 open-r1/LICENSE）
  scripts/cia/           generate_rollouts.py, build_rs_dataset.py, build_dpo_dataset.py
  src/open_r1/           sft.py, dpo.py 及其依赖
  recipes/               RS-B / DPO-A 的 yaml 与 accelerate 配置
scripts/                 打标签、评测、bootstrap 脚本
CPF_utils/               probe、B_INT / B_CoT 定义、metric、corruption 测试
cpf_evaluation.py        Hint 评测的生成入口
```

## 4. 安装

```bash
conda create -n cia python=3.11 -y && conda activate cia
pip install -r requirements.txt
```

**GPU：**
- 训练默认用 2 张卡（`NUM_GPUS`）。原实验在 80GB 的 A100、H100 和 H200 上跑。
- Hint 给 rollout 打标签时，需要额外一张卡跑 judge 服务（Qwen2.5-32B-Instruct），共 3 张。
- Hint 评测的 judge 在进程内加载，需要一张能放下 32B bf16 模型的卡。

**wandb：** recipe 默认 `report_to: wandb`。不需要的话设 `WANDB_MODE=offline`，或把 recipe 里的 `report_to` 改成 `none`。

## 5. 需要自己准备的输入（不在本目录里）

所有路径都相对于环境变量 `$SCRATCH`：

| 内容 | 路径 |
|---|---|
| base 模型权重 | `transformers/{Qwen3-8B, gemma-2-9b-it, Llama-3.1-8B-Instruct}` |
| Hint 的 judge 模型 | `transformers/Qwen2.5-32B-Instruct` |
| TwoHop 任务数据（HF DatasetDict） | `open-r1/datasets/TwoHopFact_cia_<MODEL>_linear_probe_v3_with_inner_subj` |
| TwoHop 原始 CSV（评测用） | `datasets/TwoHopFact/TwoHopFact.csv` |
| TwoHop probe | `results/open-r1/probing_results/probe_chat_filtered_trainsplit_sp/probe_chat_filtered_<MODEL>.pt` |
| Hint 任务数据（HF DatasetDict） | `open-r1/datasets/Hint_MMLU_cia` |
| Hint 原始数据（评测用） | `datasets/` 下 `cpf_evaluation.py --dataset_name Hint_MMLU` 读取的文件 |
| Hint probe | `results/open-r1/hint_mmlu_results/hint_<MODEL>_cpos_probe_v2.pt` |
| Mult prompt 数据（HF DatasetDict） | `open-r1/datasets/Mult2d_cia_<MODEL>_force_b_rdelta9_strict` |
| Mult base 模型的 corruption 文件 | `results/open-r1/math_results/2-digit-Multiplication_<MODEL>_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl` |
| Mult probe | `results/open-r1/math_results/2-digit-Multiplication_<MODEL>_8888_corruption_FIX_force_b_rdelta9_n3000_probe_TRUTH_v2.probe.pt` |

probe 目录也可以用 `TWOHOP_PROBE_DIR`、`HINT_PROBE_DIR`、`MULT_PROBE_DIR` 覆盖。

**数据划分：**
- 所有任务都是 train / validation / test = 60 / 20 / 20。
- TwoHop 用 seed 42 划分，Hint 和 Mult 用 seed 8888。
- 训练只用 train split，模型选择只看 validation，结果只报 test。
- Mult 必须用 `build_mult_probesplit_dataset.py` 生成的 `_probesplit` 划分。它和评测的 probe 划分对齐，不会泄漏。`train_offline.sh` 会自动生成它。

## 6. 训练

```bash
export SCRATCH=/path/to/your/scratch
TASK=twohop MODEL=Qwen3-8B              bash run/train_offline.sh
TASK=hint   MODEL=gemma-2-9b-it         bash run/train_offline.sh
TASK=mult   MODEL=Llama-3.1-8B-Instruct bash run/train_offline.sh
```

- 默认同时训练 RS-B 和 DPO-A，用 `VARIANTS="rsB"` 或 `VARIANTS="dpoA"` 只跑一个。
- 各任务的采样规模：TwoHop 从 train split 随机取 1800 个 prompt，最多 1024 token。Hint 用全部 1800 个 train prompt，最多 768 token。Mult 用全部 train split，最多 512 token。
- 可以设 `ROLLOUT_SEED` 做 rollout seed 复制。
- 每一步都会检查输出是否已存在，中断后重跑会接着做。
- 训练输出在 `$SCRATCH/open-r1/cia/<run>`，run 名就是 recipe 里的 `output_dir`，例如 `two_hop_qwen3_8b_rs_B_v3`、`hint_gemma_9b_full_static_dpo_A_pilot`、`multiplication_llama31_8b_static_rs_B`。

## 7. 评测

先跑 base 模型作为参照，再跑训练好的 run：

```bash
# TwoHop
MODEL=Qwen3-8B RUN=base                        bash run/eval_twohop.sh
MODEL=Qwen3-8B RUN=two_hop_qwen3_8b_rs_B_v3    bash run/eval_twohop.sh

# Hint
MODEL=gemma-2-9b-it RUN=base                                   bash run/eval_hint.sh
MODEL=gemma-2-9b-it RUN=hint_gemma_9b_full_static_dpo_A_pilot  bash run/eval_hint.sh

# Mult：先生成，再跑因果主指标
MODEL=Llama-3.1-8B-Instruct RUN=base TAG=base bash run/eval_mult_causal.sh
MODEL=Llama-3.1-8B-Instruct RUN=multiplication_llama31_8b_static_rs_B bash run/eval_mult.sh
MODEL=Llama-3.1-8B-Instruct RUN=multiplication_llama31_8b_static_rs_B TAG=final bash run/eval_mult_causal.sh
```

**统一流程：**
- 每个 checkpoint 先在 VAL 上用 gen seed 8888 评测，**只在 VAL 上选 peak**。
- peak 和最后一个 checkpoint 在 TEST 上用 3 个 gen seed（8888 / 5555 / 7777）评测。
- 解码参数：T = 0.7，top-p 0.95，seeded。
- 汇报 ΔCIA = 训练模型 − 同 seed 的 base。显著性用 paired sample-level bootstrap（1000 次）：`bash run/paired_bootstrap.sh <trained_labels.jsonl> <base_labels.jsonl>`。

**各任务的规模：**
- TwoHop 的 VAL 取 1000 条，TEST 取 2000 条。
- Hint 用 test split 里能判定的单轮样本，每个模型约 580 条。
- Mult 用 probe 划分的 TEST 行里能做 corruption 测试的 approach-B 行，每个模型约 290 到 590 条（llama 最少）。

**输出位置：** `$SCRATCH/results/open-r1/{twohop_eval_v3, hint_eval_v2, mult_eval_v2}/<run>/`。每个 `<ckpt>_<split>_s<seed>.json` 是汇总，`*labels.jsonl` 是逐条标签。

**Mult 的汇报规则：** 以因果指标为主，probe 版只作参考，两者在训练后的模型上可能方向相反。每个结果要报完整一行：macro-F1、agreement、tracked rate、B_CoT rate、四个 (B_INT, B_CoT) cell、accuracy。只有 tracked 上升、(0,1) cell 下降、accuracy 不崩这三点同时满足，才算真正的提升。只看 macro-F1 会误判：几乎全部忠实的模型会被扣分，(0,0) 退化的模型反而会得高分。

## 8. 已知注意事项

- **TwoHop 有约 4% 的重叠。** 离线 rollout 用的 1800 个 prompt 中，约 4% 也出现在评测 TEST 集里。
- **原实验在 SLURM 上跑。** 这里的脚本去掉了账号、分区和 GPU 占用相关的部分，其余步骤、参数、文件命名与原实验一致。
- **Hint 的 base 参照标签与原实验不是同一批文件。** `eval_hint.sh` 的 `RUN=base` 模式是为这个包新加的，流程与训练 run 的评测相同。原实验的 base 标签由另一个脚本生成，数值可能有细微差别。
- **没有包含的内容：**
  - GRPO（online）训练。
  - RS-A 和 DPO-B。数据构建脚本其实都支持：RS-A 用 `build_rs_dataset.py` 读打分后的文件，DPO-B 用 `build_dpo_dataset.py` 读 `_faithonly` 文件，但没有附 recipe。
  - probe 的训练代码。
  - 从原始数据构建任务数据集的代码。
  - 任何实验结果。
