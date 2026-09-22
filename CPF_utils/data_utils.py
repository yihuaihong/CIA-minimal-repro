# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under theLicense is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utility functions."""

import ast
import collections
import functools
import itertools
import sys
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from CPF_utils.model_utils import flush
import torch
from tqdm import tqdm
import os
import random
import jsonlines
from pathlib import Path


def load_dataset(dataset_name: str, dataset_dir: str='${SCRATCH}/datasets', sample_num: int = 0, seed: int = 8888):
    # Expand ${SCRATCH} / $SCRATCH etc — the default (and CLI default in
    # cpf_evaluation.py) is a literal string that must be expanded before use.
    dataset_dir = os.path.expandvars(dataset_dir)

    if dataset_name == 'TwoHopFact':
        file_path = os.path.join(dataset_dir, 'TwoHopFact/TwoHopFact.csv')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TwoHopFact 数据集文件不存在: {file_path}")

        print(f"正在加载 TwoHopFact 数据集: {file_path}")

        # 加载为 DataFrame
        df = read_dataframe(file_path)
        total_loaded = len(df)
        print(f"TwoHopFact.csv 加载完成，总样本数: {total_loaded}")

        # 采样（直接用 pandas sample，带 random_state 复现）
        if sample_num > 0 and sample_num < total_loaded:
            df = df.sample(n=sample_num, random_state=seed)
            print(f'Sampled {sample_num} samples from TwoHopFact.')
        else:
            df = df.sample(frac=1, random_state=seed)  # Shuffle 全集
            print('Running on the whole TwoHopFact dataset.')

        # 示例打印（保持原风格）
        if not df.empty:
            print('example from TwoHopFact.csv:', df.iloc[0])

        return df


    if dataset_name == 'SOCRATES':
        df = read_dataframe(os.path.join(dataset_dir, 'SOCRATES/SOCRATES_v1.csv'))
        print('example from SOCRATES_v1.csv:', df.iloc[0])
        return df

    if dataset_name == 'HoppingtooLate':
        df = read_dataframe(os.path.join(dataset_dir, 'HoppingtooLate/HoppingtooLate.csv'))
        print('example from HoppingtooLate.csv:', df.iloc[0])
        return df

    if dataset_name == 'Hint_MMLU':
        # 只加载 non-fewshot 的 False 子集（suggestion_False + posthoc_False）
        anthropic_dir = os.path.join(dataset_dir, 'antropic_faithfulness')
        if not os.path.exists(anthropic_dir):
            raise FileNotFoundError(f"Anthropic Faithfulness 数据集路径不存在: {anthropic_dir}")

        # 先找所有 False 文件，再排除包含 'fewshot' 的
        all_false_files = list(Path(anthropic_dir).glob('*_False_with_correct.jsonl'))
        jsonl_files = [f for f in all_false_files if 'fewshot' not in f.name]

        if not jsonl_files:
            raise FileNotFoundError(f"在 {anthropic_dir} 中未找到 non-fewshot 的 *_False_with_correct.jsonl 文件")

        print(f"找到 {len(jsonl_files)} 个 non-fewshot False 子数据集文件（suggestion/posthoc），正在加载...")

        all_samples = []
        for file_path in tqdm(jsonl_files, desc="加载 non-fewshot False 子数据集"):
            hint_type = file_path.stem.replace('_with_correct', '')  # e.g., suggestion_False
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    obj['hint_type'] = hint_type
                    all_samples.append(obj)

        total_loaded = len(all_samples)
        print(
            f"Hint_MMLU (non-fewshot False only) 加载完成，总样本数: {total_loaded}（来自 {len(jsonl_files)} 个子文件，预计 6000）")

        # 采样
        if sample_num > 0 and sample_num < total_loaded:
            random.seed(seed)
            all_samples = random.sample(all_samples, sample_num)
            print(f'Sampled {sample_num} samples from Hint_MMLU (non-fewshot False).')
        else:
            print('Running on the whole Hint_MMLU non-fewshot False dataset.')

        # 示例打印
        if all_samples:
            example = all_samples[0]
            print('example from Hint_MMLU (non-fewshot False):')
            print(f"  hint_type: {example.get('hint_type')}")
            print(f"  correct_answer: {example.get('correct_answer')}")
            print(f"  hint: {example.get('hint')}")
            print(
                f"  unbiased_prompt preview: {example['unbiased_prompt'][0]['content'][:300] if example['unbiased_prompt'] else 'N/A'}...")

        return all_samples

    # if dataset_name == 'Hint_GPQA':
    #     df = read_dataframe(os.path.join(dataset_dir, 'Hint_GPQA.csv'))
    #     print('example from Hint_GPQA.csv:', df.iloc[0])
    #     return df
    if dataset_name == '4-digit-Multiplication':
        file_path = os.path.join(dataset_dir, 'Multiplication/processed_valid_4digit_large.txt')
        # 如果文件在子目录，改成下面这行：
        # file_path = os.path.join(dataset_dir, '4-digit-Multiplication/processed_valid_large.txt')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"4-digit-Multiplication 数据集文件不存在: {file_path}")

        print(f"正在加载 4-digit-Multiplication 数据集: {file_path}")

        samples = []
        skipped = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="处理乘法样本"):
            line = line.strip()
            if not line:
                skipped += 1
                continue

            # 关键修复：先去除所有空白字符（空格、制表符等），得到紧凑形式如 "1338*5105"
            line_clean = ''.join(line.split())  # 移除所有空格

            # 如果不含 '*'，直接跳过（可能是无效行、行号或垃圾行）
            if '*' not in line_clean:
                skipped += 1
                continue

            # 按 '*' 分割
            parts = line_clean.split('*')
            if len(parts) != 2:
                skipped += 1
                continue

            a_str, b_str = parts[0].strip(), parts[1].strip()

            # 严格检查：必须是纯4位数字
            if not (a_str.isdigit() and b_str.isdigit() and len(a_str) == 4 and len(b_str) == 4):
                skipped += 1
                continue

            # 计算正确答案（结果可能是7~8位，无前导零）
            a = int(a_str)
            b = int(b_str)
            product = a * b
            answer_str = str(product)

            # prompt：保留原视觉形式（带空格的4位数），但用 × 替换 *
            # 先恢复带空格的原形式（从原始 line）
            original_a = ' '.join(a_str)
            original_b = ' '.join(b_str)
            prompt = f"{original_a} × {original_b} ="

            samples.append({
                "prompt": prompt,
                "answer": answer_str,
                "a": a_str,  # 无空格版本
                "b": b_str,
                "original_a_spaced": original_a,  # 带空格，用于 prompt 显示
                "original_b_spaced": original_b,
                "product": product
            })

        total_loaded = len(samples)
        print(f"4-digit-Multiplication 加载完成，总有效样本数: {total_loaded}（跳过无效行 {skipped}）")

        # 采样
        if sample_num > 0 and sample_num < total_loaded:
            random.seed(seed)
            samples = random.sample(samples, sample_num)
            print(f'Sampled {sample_num} samples from 4-digit-Multiplication.')
        else:
            print('Running on the whole 4-digit-Multiplication dataset.')

        # 示例打印
        if samples:
            example = samples[0]
            print('example from 4-digit-Multiplication:')
            print(f"  prompt: {example['prompt']}")
            print(f"  answer: {example['answer']}")
            print(f"  a (spaced): {example['original_a_spaced']}, b (spaced): {example['original_b_spaced']}")
            print(f"  product: {example['product']}")

        return samples


    if dataset_name == '3-digit-Multiplication':
        file_path = os.path.join(dataset_dir, 'Multiplication/processed_valid_3digit.txt')
        # 如果你的文件名不同，可以改成：
        # file_path = os.path.join(dataset_dir, 'processed_valid_3digit.txt')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"3-digit-Multiplication 数据集文件不存在: {file_path}")

        print(f"正在加载 3-digit-Multiplication 数据集: {file_path}")

        samples = []
        skipped = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="处理乘法样本"):
            line = line.strip()
            if not line:
                skipped += 1
                continue

            # 先去除所有空白字符，得到紧凑形式如 "133*510"
            line_clean = ''.join(line.split())

            # 如果不含 '*'，直接跳过
            if '*' not in line_clean:
                skipped += 1
                continue

            # 按 '*' 分割
            parts = line_clean.split('*')
            if len(parts) != 2:
                skipped += 1
                continue

            a_str, b_str = parts[0].strip(), parts[1].strip()

            # 严格检查：必须是纯3位数字
            if not (a_str.isdigit() and b_str.isdigit() and len(a_str) == 3 and len(b_str) == 3):
                skipped += 1
                continue

            # 计算正确答案（结果通常5~6位，无前导零）
            a = int(a_str)
            b = int(b_str)
            product = a * b
            answer_str = str(product)

            # prompt：保留原视觉形式（带空格），用 × 替换 *
            original_a = ' '.join(a_str)
            original_b = ' '.join(b_str)
            prompt = f"{original_a} × {original_b} ="

            samples.append({
                "prompt": prompt,
                "answer": answer_str,
                "a": a_str,               # 无空格版本
                "b": b_str,
                "original_a_spaced": original_a,
                "original_b_spaced": original_b,
                "product": product
            })

        total_loaded = len(samples)
        print(f"3-digit-Multiplication 加载完成，总有效样本数: {total_loaded}（跳过无效行 {skipped}）")

        # 采样
        if sample_num > 0 and sample_num < total_loaded:
            random.seed(seed)
            samples = random.sample(samples, sample_num)
            print(f'Sampled {sample_num} samples from 3-digit-Multiplication.')
        else:
            print('Running on the whole 3-digit-Multiplication dataset.')

        # 示例打印
        if samples:
            example = samples[0]
            print('example from 3-digit-Multiplication:')
            print(f"  prompt: {example['prompt']}")
            print(f"  answer: {example['answer']}")
            print(f"  a (spaced): {example['original_a_spaced']}, b (spaced): {example['original_b_spaced']}")
            print(f"  product: {example['product']}")

        return samples

    if dataset_name == '2-digit-Multiplication':
        file_path = os.path.join(dataset_dir, 'Multiplication/2digit_test.txt')
        # 如果你的文件名不同，可以改成：
        # file_path = os.path.join(dataset_dir, 'processed_valid_2digit.txt')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"2-digit-Multiplication 数据集文件不存在: {file_path}")

        print(f"正在加载 2-digit-Multiplication 数据集: {file_path}")

        samples = []
        skipped = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="处理乘法样本"):
            line = line.strip()
            if not line:
                skipped += 1
                continue

            # 先去除所有空白字符，得到紧凑形式如 "13*51"
            line_clean = ''.join(line.split())

            # 如果不含 '*'，直接跳过
            if '*' not in line_clean:
                skipped += 1
                continue

            # 按 '*' 分割
            parts = line_clean.split('*')
            if len(parts) != 2:
                skipped += 1
                continue

            a_str, b_str = parts[0].strip(), parts[1].strip()

            # 严格检查：必须是纯2位数字
            if not (a_str.isdigit() and b_str.isdigit() and len(a_str) == 2 and len(b_str) == 2):
                skipped += 1
                continue

            # 计算正确答案（结果通常3~4位，无前导零）
            a = int(a_str)
            b = int(b_str)
            product = a * b
            answer_str = str(product)

            # prompt：保留原视觉形式（带空格），用 × 替换 *
            original_a = ' '.join(a_str)
            original_b = ' '.join(b_str)
            prompt = f"{original_a} × {original_b} ="

            samples.append({
                "prompt": prompt,
                "answer": answer_str,
                "a": a_str,
                "b": b_str,
                "original_a_spaced": original_a,
                "original_b_spaced": original_b,
                "product": product
            })

        total_loaded = len(samples)
        print(f"2-digit-Multiplication 加载完成，总有效样本数: {total_loaded}（跳过无效行 {skipped}）")

        # 采样
        if sample_num > 0 and sample_num < total_loaded:
            random.seed(seed)
            samples = random.sample(samples, sample_num)
            print(f'Sampled {sample_num} samples from 2-digit-Multiplication.')
        else:
            print('Running on the whole 2-digit-Multiplication dataset.')

        # 示例打印
        if samples:
            example = samples[0]
            print('example from 2-digit-Multiplication:')
            print(f"  prompt: {example['prompt']}")
            print(f"  answer: {example['answer']}")
            print(f"  a (spaced): {example['original_a_spaced']}, b (spaced): {example['original_b_spaced']}")
            print(f"  product: {example['product']}")

        return samples





def convert_object(x: Any) -> Any:
  """Convert a string representation of a list or dict to an actual list or dict."""
  if pd.isnull(x):
    return np.nan
  if x in ['', 'nan']:
    return np.nan
  if isinstance(x, str):
    return ast.literal_eval(x)
  return x


def convert_correct(x: Any) -> bool | float:
  """Convert a string representation of a boolean to an actual boolean."""
  if pd.isnull(x):
    return np.nan
  if x in ['', 'nan']:
    return np.nan
  if isinstance(x, str):
    return ast.literal_eval(x)
  if isinstance(x, bool):
    return x
  raise ValueError(f'Unknown correct value: {x}')


# def read_dataframe(
#     path: str,
#     eval_cols: list[str] = (
#         '.aliases',
#         '.list',
#         '.entities',
#         '.keywords',
#         '.matches',
#     ),
#     keep_default_na: bool = False,
# ) -> pd.DataFrame:
#   """Read a CSV file into a pandas DataFrame and convert specific columns.
#
#   Args:
#       path: The path to the CSV file.
#       eval_cols: List of column suffixes to evaluate.
#       keep_default_na: Whether to keep default NaN values.
#
#   Returns:
#       A pandas DataFrame with converted columns.
#   """
#   df = pd.read_csv(path, keep_default_na=keep_default_na)
#   for col in df.columns:
#     if any([col.endswith(ec) for ec in eval_cols]):
#       df.loc[:, col] = df[col].apply(convert_object)
#
#   for e in ['e1', 'e2', 'e3']:
#     if f'{e}.value' in df:
#       df.loc[:, f'{e}.value'] = df[f'{e}.value'].astype(str)
#
#   count_cols = [
#       col
#       for col in df
#       if any(
#           subs in col.split('.')[-1]
#           for subs in ['count', 'c4', 'dolma', 'oscar', 'openwebtext']
#       )
#   ]
#   for col in count_cols:
#     df.loc[:, col] = df[col].apply(
#         lambda x: int(float(x))
#         if (x not in ['', 'nan']) and (not pd.isnull(x))
#         else np.nan
#     )
#
#   correct_cols = [col for col in df if 'correct' in col.split('.')[-1]]
#   for col in correct_cols:
#     df.loc[:, col] = df[col].apply(convert_correct)
#
#   if 'tid' in df:
#     df = df.sort_values(by='tid')
#   if 'eid' in df:
#     df = df.sort_values(by='eid').reset_index(drop=True)
#
#   return df


def read_dataframe(
    path: str,
    eval_cols: list[str] = (
        '.aliases',
        '.list',
        '.entities',
        '.keywords',
        '.matches',
    ),
    keep_default_na: bool = False,
) -> pd.DataFrame:
  """Read a CSV file into a pandas DataFrame and convert specific columns.

  Args:
      path: The path to the CSV file.
      eval_cols: List of column suffixes to evaluate.
      keep_default_na: Whether to keep default NaN values.

  Returns:
      A pandas DataFrame with converted columns.
  """
  df = pd.read_csv(path, keep_default_na=keep_default_na)
  for col in df.columns:
    if any([col.endswith(ec) for ec in eval_cols]):
      df[col] = df[col].astype('object')  # 添加这一行，将 dtype 转换为 'object'
      df.loc[:, col] = df[col].apply(convert_object)

  for e in ['e1', 'e2', 'e3']:
    if f'{e}.value' in df:
      df.loc[:, f'{e}.value'] = df[f'{e}.value'].astype(str)

  count_cols = [
      col
      for col in df
      if any(
          subs in col.split('.')[-1]
          for subs in ['count', 'c4', 'dolma', 'oscar', 'openwebtext']
      )
  ]
  for col in count_cols:
    df.loc[:, col] = df[col].apply(
        lambda x: int(float(x))
        if (x not in ['', 'nan']) and (not pd.isnull(x))
        else np.nan
    )

  correct_cols = [col for col in df if 'correct' in col.split('.')[-1]]
  for col in correct_cols:
    df.loc[:, col] = df[col].apply(convert_correct)

  if 'tid' in df:
    df = df.sort_values(by='tid')
  if 'eid' in df:
    df = df.sort_values(by='eid').reset_index(drop=True)

  return df


def get_efficient_batchified_info(
    df: pd.DataFrame, param_to_column: dict[str, str]
) -> tuple[list[list[int]], dict[str, list[Any]]]:
  """Get efficiently batchified information for efficient processing.

  Args:
      df: The input DataFrame.
      param_to_column: A dictionary mapping parameter names to column names.

  Returns:
      A tuple containing batched indices and batched parameter values.
  """
  columns = list(param_to_column.values())
  params = list(param_to_column.keys())

  assert df.index.is_unique

  subdfs = []
  inputs = []
  for values, subdf in tqdm.tqdm(df.groupby(columns), desc='making batches'):
    inputs.append(values)
    subdfs.append(subdf)
  inputs = list(zip(*inputs))

  param_to_column_values = {k: v for k, v in zip(params, inputs)}

  # sort inputs and subdfs by max length of inputs
  lengths = [len(x) for x in param_to_column_values[params[0]]]
  sorted_indices = np.argsort(lengths, kind='stable')[::-1]

  batched_param_to_column_values = {
      param: [param_to_column_values[param][i] for i in sorted_indices]
      for param in params
  }
  batched_indices = [subdfs[i].index.tolist() for i in sorted_indices]

  return batched_indices, batched_param_to_column_values


@torch.no_grad()
def efficient_batchify(
    df: pd.DataFrame,
    param_to_column: dict[str, str],
    function: Callable[..., Any],
    batch_size: int = 4,
    max_size: Optional[int] = None,
    tqdm_desc: Optional[str] = '',
    flush_step: Optional[int] = None,
    concat_dim: Optional[int] = None,
) -> Callable[..., Any]:
  """Efficiently batchify a function for processing a DataFrame.

  Args:
      df: The input DataFrame.
      param_to_column: A dictionary mapping parameter names to column names.
      function: The function to batchify.
      batch_size: The batch size for processing.
      max_size: The maximum size for processing.
      tqdm_desc: The description for the tqdm progress bar.
      flush_step: The step interval for flushing.
      concat_dim: The dimension for concatenation.

  Returns:
      A batchified function.
  """
  indices, param_to_column_values = get_efficient_batchified_info(
      df, param_to_column
  )

  @functools.wraps(function)
  def batchified_function(**kwargs):
    suboutputs = batchify(
        function,
        batch_size=batch_size,
        max_size=max_size,
        tqdm_desc=tqdm_desc,
        flush_step=flush_step,
        concat_dim=concat_dim,
    )(param_to_column_values, **kwargs)

    return unrolled_outputs(suboutputs, indices, concat_dim)

  return batchified_function


def unrolled_outputs(
    suboutputs: Any, batched_indices: list[list[int]], concat_dim: Optional[int]
) -> Any:
  """Unroll batched outputs to match the original indices.

  Args:
      suboutputs: The batched outputs.
      batched_indices: The batched indices.
      concat_dim: The dimension for concatenation.

  Returns:
      The unrolled outputs.
  """
  outputs = dict()
  if isinstance(suboutputs, (list, tuple)):
    for suboutput, indices in zip(suboutputs, batched_indices):
      for index in indices:
        outputs[index] = suboutput
  elif isinstance(suboutputs, dict):
    for k, subout in suboutputs.items():
      outs = unrolled_outputs(subout, batched_indices, concat_dim)
      outputs[k] = outs
  elif isinstance(suboutputs, torch.Tensor):
    if concat_dim != 0:
      suboutputs = suboutputs.transpose(0, concat_dim)
    for suboutput, indices in zip(suboutputs, batched_indices):
      for index in indices:
        outputs[index] = suboutput
  else:
    raise NotImplementedError
  return outputs


@torch.no_grad()
def batchify(
    function: Callable[..., Any],
    batch_size: int = 4,
    max_size: Optional[int] = None,
    tqdm_desc: Optional[str] = '',
    concat_dim: Optional[int] = None,
    flush_step: Optional[int] = None,
) -> Callable[..., Any]:
  """Batchify a function for processing.

  Args:
      function: The function to batchify.
      batch_size: The batch size for processing.
      max_size: The maximum size for processing.
      tqdm_desc: The description for the tqdm progress bar.
      concat_dim: The dimension for torch tensor concatenation.
      flush_step: The step interval for flushing.

  Returns:
      A batchified function.
  """

  @functools.wraps(function)
  def batchified_function(
      inputs: dict[str, list[Any]] | list[Any], **kwargs
  ) -> Any:
    results = []

    if isinstance(inputs, dict):
      upper_bound = len(inputs[list(inputs.keys())[0]])
    else:
      upper_bound = len(inputs)

    if upper_bound % batch_size > 0:
      upper_bound = (upper_bound // batch_size + 1) * batch_size

    iter_step = 0
    for start in tqdm.tqdm(
        range(0, upper_bound, batch_size),
        file=sys.stdout,
        disable=True if tqdm_desc is None else False,
        desc=tqdm_desc,
    ):
      try:
        if max_size and start > max_size:
          break

        if start + batch_size > upper_bound:
          local_batch_size = upper_bound - start
        else:
          local_batch_size = batch_size

        if isinstance(inputs, dict):
          batched_inputs = {
              k: v[start : start + local_batch_size] for k, v in inputs.items()
          }
          results.append(function(**batched_inputs, **kwargs))
        else:
          batched_inputs = inputs[start : start + local_batch_size]
          results.append(function(batched_inputs, **kwargs))

        iter_step += 1
        if flush_step is not None and iter_step % flush_step == 0:
          flush()
      except KeyboardInterrupt:
        print('KeyboardInterrupt at iter_step', iter_step)
        break

    return aggregated_results(results, concat_dim)

  return batchified_function


def aggregated_results(results: list[Any], concat_dim: Optional[int]) -> Any:
  """Aggregate batched results.

  Args:
      results: The batched results.
      concat_dim: The dimension for concatenation.

  Returns:
      The aggregated results.
  """
  # flatten the result
  if isinstance(results[0], torch.Tensor):
    results = torch.cat(results, dim=concat_dim)
  elif isinstance(results[0], (list, tuple)):
    if concat_dim is not None and isinstance(results[0][0], torch.Tensor):
      # assert isinstance(results[0][0], torch.Tensor)
      outputs = [[] for _ in range(len(results[0]))]
      for result in results:
        for i, r in enumerate(result):
          outputs[i].append(r)
      for i in range(len(outputs)):
        outputs[i] = torch.cat(outputs[i], dim=concat_dim)
      results = outputs
    else:
      results = list(itertools.chain.from_iterable(results))
  elif isinstance(results[0], dict):
    new_results = collections.defaultdict(list)
    keys = results[0].keys()
    for k in keys:
      v = [result[k] for result in results]
      v = aggregated_results(v, concat_dim)
      new_results[k] = v
    results = dict(new_results)
  else:
    raise NotImplementedError
  return results
