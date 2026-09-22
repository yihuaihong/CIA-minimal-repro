# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the functions for running the experiments in the paper "Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?"."""

import functools
import gc
import itertools
import re
import string
from typing import Any
import torch
import jsonlines
from pathlib import Path
from tqdm import tqdm
import re
import json
import os



import pandas as pd
from CPF_utils import model_utils
from CPF_utils.data_utils import batchify
from CPF_utils.model_utils import is_instruction_tuned
from CPF_utils.tokenization_utils import get_completion
import torch
from tqdm import tqdm
import transformers
import unidecode
from CPF_utils.data_utils import load_dataset
from CPF_utils.logitlens_utils import run_logit_lens_evaluation, run_attn_lens_evaluation #, run_attribution_graph_evaluation
from CPF_utils.probing_utils import run_two_hop_linear_probe_evaluation #, run_attribution_graph_evaluation
from CPF_utils.gemini_caller_utils import HintAIAgent

# from vllm import SamplingParams


# fact types for pretrained models
pretrained_fact_types = {
    "condition": "r1(e1)",
    "base": "r2(e2)",
    "entailed": "r2(r1(e1))",
    "entailed.e2.null": "r2(e2.null)",
    "entailed.e1.null": "r2(r1(e1.null))",
}

# fact types for instruction-tuned models
it_fact_types = {
    "blank.cot": "r2(r1(e1)).blank.cot",
    "blank.entailed": "r2(r1(e1)).blank",
    "blank.condition": "r1(e1).blank",
    "blank.base": "r2(e2).blank",
    "blank.e2.null": "r2(e2.null).blank",
    "blank.e1.null": "r2(r1(e1.null)).blank",
}

# answer entity types for fact types
answer_entity_types = {
    "r1(e1)": "e2",
    "r2(e2)": "e3",
    "r2(r1(e1))": "e3",
    "r2(r1(e1)).appositive": "e2",
    "r2(e2.null)": "e3",
    "r2(r1(e1.null))": "e3",
    "r2(e1)": "e3",
    "r2(r1(e1)).cot": "e3",
    "r1(e1).blank": "e2",
    "r2(e2).blank": "e3",
    "r2(r1(e1)).blank": "e3",
    "r2(r1(e1)).blank.cot": "e3",
    "r2(e2.null).blank": "e3",
    "r2(r1(e1.null)).blank": "e3",
    "r2(r1(e1)).hint_think": "e3",
}


def generate_completion_answers(questions, model, tokenizer, n_new_tokens=100, do_sample=False):
    inputs = tokenizer(questions, return_tensors="pt", padding="longest", return_token_type_ids=False).to('cuda')
    input_length = inputs.input_ids.size(1)
    gen_tokens = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=do_sample)

    gen_text = tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)

    return gen_text

# def generate_chat_answers(questions, model, tokenizer, n_new_tokens=100, do_sample=False, do_thinking=False, add_generation_prompt=False):
#
#     Question_input = [[{"role": "user", "content": prompt}] for prompt in questions]
#     # 可以试试有没有add_generation_prompt有什么区别
#     if 'qwen3' in model.config.model_type.lower():
#         if do_thinking:
#             texts = tokenizer.apply_chat_template(Question_input ,tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=True)
#             print('texts[0] in thinking: ',texts[0])
#         else:
#             texts = tokenizer.apply_chat_template(Question_input ,tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
#             print('texts[0] in no thinking: ', texts[0])
#     else:
#         texts = tokenizer.apply_chat_template(Question_input, tokenize=False, add_generation_prompt=add_generation_prompt)
#     #texts = tokenizer.apply_chat_template(Question_input ,add_generation_prompt = True, tokenize = False)
#     inputs = tokenizer(texts, padding="longest", return_tensors="pt")
#     inputs = {key: val.cuda() for key, val in inputs.items()}
#     temp_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
#     gen_tokens = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=do_sample)
#     gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
#     gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]
#
#     return gen_text


def extract_answer(gen_text):
    """
    从模型生成的文本中提取最终答案字母 (A-D)。
    优先匹配 <mc>X</mc>，否则匹配最后一个 (X) 或 X.
    """
    # 优先找 <mc>X</mc>
    mc_match = re.search(r'<mc\s*>\s*([A-D])\s*</mc\s*>', gen_text, re.IGNORECASE)
    if mc_match:
        return mc_match.group(1).upper()

    # 其次找最后一个 (X)
    paren_matches = re.findall(r'\(([A-D])\)', gen_text.upper())
    if paren_matches:
        return paren_matches[-1]

    # 最后尝试直接找 A、B、C、D（可能出现在句子末尾）
    letter_match = re.search(r'\b([A-D])\b', gen_text.upper())
    if letter_match:
        return letter_match.group(1)

    return None  # 无法解析


def generate_chat_answers(
        questions,
        model,
        tokenizer,
        n_new_tokens=512,
        # do_sample=False,
        do_sample: bool = True,  # 改为 True 以启用随机采样
        temperature: float = 0.7,  # 引入温度，值在 0.1-1.0 间，较低更“保守”
        top_p: float = 0.95,  # Nucleus sampling，保留累积概率 > top_p 的 token
        top_k: int = 50,  # 只从 top_k 个最可能 token 中采样
        do_thinking=False,
        add_generation_prompt=True,
        debug=False,
        seed: int = None,  # 可选传入 seed
):
    if seed is not None:
        torch.manual_seed(seed)  # 设置 seed 以控制随机性

    # 支持两种输入：list[str] 或 list[list[dict]]
    if isinstance(questions[0], str):
        chat_inputs = [[{"role": "user", "content": q}] for q in questions]
    else:
        chat_inputs = questions  # 多轮消息列表

    # === 关键修复：Anthropic 数据集使用 "human" role，而 Transformers 标准是 "user" ===
    # 将所有 "human" 改为 "user"（"assistant" 保持不变）
    standardized_chats = []
    for chat in chat_inputs:
        standardized_chat = []
        for msg in chat:
            role = msg["role"]
            if role == "human":
                role = "user"  # 映射为标准 role
            standardized_chat.append({"role": role, "content": msg["content"]})
        standardized_chats.append(standardized_chat)

    # 统一 apply_chat_template
    apply_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if do_thinking and hasattr(tokenizer, "enable_thinking"):
        apply_kwargs["enable_thinking"] = True
    else:
        apply_kwargs["enable_thinking"] = False

    texts = tokenizer.apply_chat_template(standardized_chats, **apply_kwargs)

    if debug and texts:
        print("Sample formatted prompt:", texts[0][:1000])

    inputs = tokenizer(texts, padding="longest", return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_tokens = model.generate(
            **inputs,
            max_new_tokens=n_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 更精确截取新生成部分
    generated = gen_tokens[:, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return gen_text


def get_explanation(text):
    parts = text.split("**Explanation:**", 1)
    return parts[1].strip() if len(parts) > 1 else None

def detect_answer(answer, ground_truth):

    # 将文本和答案都转换为小写，确保大小写不敏感
    return ground_truth.lower() in answer.lower()

def get_completion_messages(
    prompt: str, prompt_key: str, model_name: str
) -> list[dict[str, str]]:
  """Get the completion messages for the prompt.

  The messages are based on the prompt key and model name for instruction-tuned
  models.

  Args:
      prompt: The prompt text.
      prompt_key: The key indicating the type of prompt.
      model_name: The name of the model.

  Returns:
      A list of dictionaries containing the role and content of the messages.
  """
  if "think" in prompt_key:
    instruction = (
        "Fill in the blank. Write down only what goes in the blank. Think"
        " step-by-step, but do it only internally and do not explain it in the"
        " answer. The answer can consist of multiple words."
    )
    separator = "\n\n"
  elif "blank" in prompt_key:
    if prompt_key.endswith("cot.prompt"):
      instruction = (
          "Fill in the blank. First, write the step-by-step explanation"
          ' necessary to get the solution with the prefix "EXPLANATION:". After'
          ' that, write down the final answer with the prefix "ANSWER:". For'
          " the final answer, write down only what goes in the blank. The"
          " answer can consist of multiple words."
      )
    else:
      instruction = (
          "Fill in the blank. Write down only what goes in the blank. Do not"
          " explain your answer. The answer can consist of multiple words."
      )
    separator = "\n\n"
  else:
    raise ValueError(f"Unknown prompt key: {prompt_key}")

  start_role = model_utils.get_messages_start_role(model_name)

  if start_role == "user":
    return [
        {"role": "user", "content": f"{instruction}{separator}{prompt}"},
    ]
  else:
    return [
        {"role": start_role, "content": instruction},
        {"role": "user", "content": prompt},
    ]


# def get_input_dict(
#     tokenizer: Any, prompt_key: str, prompts: list[str], instruction_tuned: bool
# ) -> dict[str, Any]:
#   """Get the input dictionary for the vLLm models based on the prompts and whether the model is instruction-tuned.
#
#   Args:
#       tokenizer: The tokenizer to use.
#       prompt_key: The key indicating the type of prompt.
#       prompts: The list of prompts.
#       instruction_tuned: Whether the model is instruction-tuned.
#
#   Returns:
#       A dictionary containing the input prompts and sampling parameters.
#   """
#   input_dict = dict()
#   if instruction_tuned:
#     assert (
#         tokenizer.chat_template
#     ), "Instruction-tuned models require chat templates"
#     print("Using chat template")
#     messages = [
#         get_completion_messages(prompt, prompt_key, tokenizer.name_or_path)
#         for prompt in prompts
#     ]
#     input_dict["prompts"] = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#   else:
#     input_dict["prompts"] = prompts
#
#   input_dict["sampling_params"] = get_vllm_sampling_params(
#       prompt_key, instruction_tuned
#   )
#
#   return input_dict


# def get_vllm_sampling_params(
#     prompt_key: str, instruction_tuned: bool
# ) -> SamplingParams:
#   """Get the sampling parameters for the vLLM model based on the prompt key and whether the model is instruction-tuned.
#
#   Args:
#       prompt_key: The key indicating the type of prompt.
#       instruction_tuned: Whether the model is instruction-tuned.
#
#   Returns:
#       The sampling parameters.
#   """
#   params = {"seed": 0}
#
#   if "blank" in prompt_key:  # instruction-tuned models
#     if "cot" in prompt_key:
#       params["max_tokens"] = 512
#     else:
#       params["max_tokens"] = 96
#   elif (
#       instruction_tuned
#   ):  # instruction-tuned models might repeat the input,
#     # so we need to increase the max tokens
#     params["max_tokens"] = 96
#   else:
#     params["max_tokens"] = 32
#
#   params["temperature"] = 0  # greedy decoding without randomness
#
#   return SamplingParams(**params)


# def run_vllm_completion(
#     llm: Any,
#     df: pd.DataFrame,
#     fact_type: str,
#     instruction_tuned: bool,
#     force_completion: bool = False,
# ) -> pd.DataFrame:
#   """Run prompt completion with vLLM models for the given DataFrame and composition type.
#
#   Args:
#       llm: The vLLM model.
#       df: The input DataFrame.
#       fact_type: The fact composition type.
#       instruction_tuned: Whether the model is instruction-tuned.
#       force_completion: Whether to force completion even if it already exists.
#
#   Returns:
#       The updated DataFrame with completions.
#   """
#   if force_completion or (f"{fact_type}.completion" not in df):
#     assert len(df) == len(df["uid"].unique())
#
#     prompt_key = f"{fact_type}.prompt"
#
#     # If there are many same prompts, group by prompt and generate
#     # completions for each group
#     prompt_to_uids = (
#         df.groupby(prompt_key).apply(lambda x: set(x["uid"].tolist())).to_dict()
#     )
#     prompts = sorted(set(prompt_to_uids.keys()), key=len, reverse=True)
#     outputs = llm.generate(
#         **get_input_dict(
#             llm.get_tokenizer(), prompt_key, prompts, instruction_tuned
#         )
#     )
#
#     update_df_with_completion(df, prompts, outputs, prompt_to_uids, fact_type)
#
#   return df


# def run_hf_completion(
#     model: Any,
#     tokenizer: Any,
#     df: pd.DataFrame,
#     fact_type: str,
#     instruction_tuned: bool,
#     batch_size: int = 4,
#     force_completion: bool = False,
# ) -> pd.DataFrame:
#   """Run prompt completion with HuggingFace models for the given DataFrame and composition type.
#
#   Args:
#       model: The Hugging Face model.
#       tokenizer: The tokenizer to use.
#       df: The input DataFrame.
#       fact_type: The fact composition type.
#       instruction_tuned: Whether the model is instruction-tuned.
#       batch_size: The batch size for processing.
#       force_completion: Whether to force completion even if it already exists.
#
#   Returns:
#       The updated DataFrame with completions.
#   """
#   if force_completion or (f"{fact_type}.completion" not in df):
#     assert len(df) == len(df["uid"].unique())
#
#     completions = batchify(
#         get_hf_completions,
#         batch_size=batch_size,
#     )(
#         {"prompts": df[f"{fact_type}.prompt"].tolist()},
#         model=model,
#         tokenizer=tokenizer,
#         prompt_key=f"{fact_type}.prompt",
#         instruction_tuned=instruction_tuned,
#     )
#
#     df.loc[:, f"{fact_type}.completion"] = completions
#
#   return df


# def get_hf_completions(
#     model: Any,
#     tokenizer: Any,
#     prompt_key: str,
#     prompts: list[str],
#     instruction_tuned: bool,
# ) -> list[str]:
#   """Get HuggingFace model completions for the given prompts.
#
#   Args:
#       model: The Hugging Face model.
#       tokenizer: The tokenizer to use.
#       prompt_key: The key indicating the type of prompt.
#       prompts: The list of prompts.
#       instruction_tuned: Whether the model is instruction-tuned.
#
#   Returns:
#       The list of completions.
#   """
#   input_dict = get_input_dict(
#       tokenizer,
#       prompt_key,
#       prompts,
#       instruction_tuned,
#   )
#   prompts = input_dict["prompts"]
#   new_max_tokens = input_dict["sampling_params"].max_tokens
#
#   assert tokenizer.padding_side == "left"
#   prompt_inputs = tokenizer(
#       prompts, return_tensors="pt", padding=True, truncation=True
#   ).to(model.device)
#
#   if "Nemo-Base" in model.name_or_path:
#     del prompt_inputs["token_type_ids"]
#
#   completions = model.generate(
#       **prompt_inputs,
#       pad_token_id=tokenizer.eos_token_id,
#       max_new_tokens=new_max_tokens,
#       do_sample=False,
#   )
#   return get_completion(completions, prompt_inputs, tokenizer)


def update_df_with_completion(
    df: pd.DataFrame,
    prompts: list[str],
    outputs: Any,
    prompt_to_uids: dict[str, list[int]],
    fact_type: str,
) -> None:
  """Update the DataFrame with completions.

  Args:
      df: The input DataFrame.
      prompts: The list of prompts.
      outputs: The generated outputs.
      prompt_to_uids: A dictionary mapping prompts to UIDs.
      fact_type: The fact composition type.
  """
  results = []
  for prompt, output in tqdm.tqdm(
      zip(prompts, outputs),
      total=len(prompts),
      desc=f"Extracting {fact_type} completions",
  ):
    uids = prompt_to_uids[prompt]

    completion = output.outputs[0].text
    results.extend([(uid, completion) for uid in uids])

  # Create DataFrame from results
  new_columns = [f"{fact_type}.completion"]

  for column in new_columns:
    if column in df:
      df.drop(columns=[column], inplace=True)

  results_df = pd.DataFrame(results, columns=["uid"] + new_columns)
  merged = df.merge(results_df, on="uid", how="left")

  completions = merged[f"{fact_type}.completion"].tolist()
  df.insert(
      df.columns.get_loc(f"{fact_type}.prompt") + 1,
      f"{fact_type}.completion",
      completions,
  )


def shortcut_free_evaluate(
    df: pd.DataFrame,
    fact_type: str,
    answer_entity_type: str,
    answer_postfix: str = "aliases",
    normalize: bool = True,
) -> pd.DataFrame:
  """Evaluate the completions for a fact_type in a shortcut-free manner.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
      answer_entity_type: The answer entity type.
      answer_postfix: The postfix for the answer column.
      normalize: Whether to normalize the text.

  Returns:
      The updated DataFrame with evaluation results.
  """
  matches_col = "matches" if normalize else "strict.matches"
  correct_col = "correct" if normalize else "strict.correct"
  inst_col = "failed_instruction" if normalize else "strict.failed_instruction"

  df.loc[:, f"{fact_type}.completion"] = df.loc[
      :, f"{fact_type}.completion"
  ].astype(str)

  df.loc[:, f"{fact_type}.{matches_col}"] = df.apply(
      lambda row: get_matches(
          row[f"{fact_type}.completion"],
          row[f"{answer_entity_type}.{answer_postfix}"],
          normalize=normalize,
      ),
      axis=1,
  )
  df.loc[:, f"{fact_type}.{correct_col}"] = df.loc[
      :, f"{fact_type}.{matches_col}"
  ].apply(lambda x: len(x) > 0)  # pylint: disable=g-explicit-length-test

  if fact_type.startswith(("r1(e1)", "r2(e2)")):
    fill_has_multiple_choice_format(df, fact_type)
    condition = (
        df[f"{fact_type}.{correct_col}"]
        & df[f"{fact_type}.has_multiple_choice_format"]
    )
    df.loc[condition, f"{fact_type}.{correct_col}"] = False

  if fact_type not in ["r2(r1(e1))", "r2(r1(e1)).blank"]:
    return df

  completion_col = "completion"

  # Direct assignment by expanding the tuples/lists returned by apply
  results = df.apply(
      lambda row: get_real_correct_and_failed_instruction(
          row, fact_type, completion_col, normalize=normalize
      ),
      axis=1,
  )

  # Separate results into the respective columns
  df.loc[:, f"{fact_type}.real.{correct_col}"] = results.apply(lambda x: x[0])
  df.loc[:, f"{fact_type}.{inst_col}"] = results.apply(lambda x: x[1])

  df.loc[:, f"{fact_type}.real.{correct_col}"] = df[
      f"{fact_type}.real.{correct_col}"
  ].astype(bool)
  df.loc[:, f"{fact_type}.{inst_col}"] = df[f"{fact_type}.{inst_col}"].astype(
      bool
  )

  print(f"set unusable for {fact_type}")
  fill_has_multiple_choice_format(df, fact_type)
  condition = (
      df[f"{fact_type}.{correct_col}"]
      & df[f"{fact_type}.has_multiple_choice_format"]
  )
  df.loc[condition, f"{fact_type}.real.{correct_col}"] = False
  unusable_col = f"{fact_type}.unusable"

  unusable_condition = (
      df[f"{fact_type}.has_multiple_choice_format"]
      | df[f"{fact_type}.{inst_col}"]
  ) & df[f"{fact_type}.{correct_col}"]

  df.loc[:, unusable_col] = False
  df.loc[unusable_condition, unusable_col] = True

  df.loc[:, unusable_col] = df[unusable_col].astype(bool)

  return df


def get_real_correct_and_failed_instruction(
    row: pd.Series,
    fact_type: str,
    completion_postfix: str,
    normalize: bool = True,
) -> tuple[bool, bool]:
  """Get the real correct (set to False when e2 is generated before e3) and failed instruction status for the given row.

  Args:
      row: The input row.
      fact_type: The fact composition type.
      completion_postfix: The postfix for the completion column.
      normalize: Whether to normalize the text.

  Returns:
      A tuple containing the real correct status and failed instruction status.
  """
  assert completion_postfix in ["completion", "real.completion"]

  correct_col = "correct" if normalize else "strict.correct"

  if not row[f"{fact_type}.{correct_col}"]:
    return False, False

  completion = row[f"{fact_type}.{completion_postfix}"]
  if not completion:
    return False, False

  if normalize:
    completion = normalize_text(completion)

  e2_aliases = list(set(itertools.chain.from_iterable(row["e2.aliases"])))
  e3_aliases = list(set(itertools.chain.from_iterable(row["e3.aliases"])))

  if normalize:
    e2_aliases = list(set([normalize_text(alias) for alias in e2_aliases]))
    e3_aliases = list(set([normalize_text(alias) for alias in e3_aliases]))

    e2_aliases = [alias for alias in e2_aliases if alias]
    e3_aliases = [alias for alias in e3_aliases if alias]

  e2_indices = [completion.find(e2) for e2 in e2_aliases]
  e3_indices = [completion.find(e3) for e3 in e3_aliases]

  assert e3_indices

  e2_valid_indices = [idx for idx in e2_indices if idx != -1]
  e3_valid_indices = [idx for idx in e3_indices if idx != -1]

  if not e2_valid_indices:
    return True, False

  min_e2_index = min(e2_valid_indices)
  min_e3_index = min(e3_valid_indices)

  if min_e2_index < min_e3_index:
    return False, True

  return True, False


def fill_has_multiple_choice_format(df: pd.DataFrame, fact_type: str) -> None:
  """Fill the DataFrame with a column indicating whether the completion has the form of a multiple-choice question.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
  """
  has_multiple_choice_format = df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["A.", "B.", "C."])
  )
  has_multiple_choice_format |= df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["A)", "B)", "C)"])
  )
  has_multiple_choice_format |= df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["1.", "2.", "3."])
  )
  has_multiple_choice_format |= df[f"{fact_type}.completion"].apply(
      lambda x: all(choice in x for choice in ["1)", "2)", "3)"])
  )
  df.loc[:, f"{fact_type}.has_multiple_choice_format"] = (
      has_multiple_choice_format
  )


def run_shortcut_free_evaluation(
    df: pd.DataFrame, normalize: bool = True, force: bool = False
) -> None:
  """Run shortcut-free evaluation on the DataFrame.

  Args:
      df: The input DataFrame.
      normalize: Whether to normalize the text.
      force: Whether to force evaluation even if it already exists.
  """
  if normalize:
    correct_col = "correct"
  else:
    correct_col = "strict.correct"

  for col in df.columns:
    if col.endswith(".real.completion"):
      continue

    if col.endswith(".completion"):
      fact_type = col.rsplit(".", 1)[0]

      if fact_type not in list(pretrained_fact_types.values()) + list(
          it_fact_types.values()
      ):
        continue

      if not force and f"{fact_type}.{correct_col}" in df:
        continue

      print(fact_type)
      answer_entity_type = answer_entity_types[fact_type]
      df = shortcut_free_evaluate(
          df, fact_type, answer_entity_type, normalize=normalize
      )


def transform_punctuation(text: str) -> str:
  """Replace punctuation in the text with spaces.

  Args:
      text: The input text.

  Returns:
      The text with punctuation replaced by spaces.
  """
  translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
  text = text.translate(translator)
  return text


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_accents: bool = True,
    remove_articles: bool = True,
    remove_spaces_in_abbr: bool = True,
    remove_punctuations: bool = True,
    remove_spaces: bool = False,
) -> str:
  """Normalize the input text.

  Args:
      text: The input text.
      lowercase: Whether to convert the text to lowercase.
      remove_accents: Whether to remove accents from the text.
      remove_articles: Whether to remove articles from the text.
      remove_spaces_in_abbr: Whether to remove spaces in abbreviations.
      remove_punctuations: Whether to remove punctuations from the text.
      remove_spaces: Whether to remove all spaces from the text.

  Returns:
      The normalized text.
  """
  text = text.strip()

  if remove_spaces_in_abbr:
    text = re.sub(r"(?<=\b[A-Z])\. (?=[A-Z]\.)", ".", text)

  if lowercase:
    text = text.lower()

  # remove accents
  if remove_accents:
    text = unidecode.unidecode(text)

  # remove articles
  if remove_articles:
    text = re.sub(r"\b(the|an|a)\b(?=\s)", "", text, flags=re.IGNORECASE)

  # replace punctuation with spaces
  if remove_punctuations:
    text = transform_punctuation(text)

  # replace multiple spaces with single space
  text = re.sub(r"\s+", " ", text)

  # remove all spaces
  if remove_spaces:
    text = text.replace(" ", "")

  return text.strip()


def get_matches(
    completion: str, correct_answers: list[list[str]], normalize: bool = True
) -> tuple[str, ...]:
  """Get a subset of correct_answers that appear in the completion.

  Args:
      completion: The generated completion.
      correct_answers: The list of correct answers.
      normalize: Whether to normalize the text.

  Returns:
      A tuple of the matched answers that appear in the completion.
  """
  assert isinstance(correct_answers, (list, tuple)), correct_answers
  if correct_answers:
    assert isinstance(correct_answers[0], (list, tuple)), correct_answers[0]

  completion = completion.strip()
  if normalize:
    completion = normalize_text(completion)

  matched_answers = []
  for answers in correct_answers:
    for possible_answer in answers:
      possible_answer = f"{possible_answer}"

      possible_answer_to_compare = possible_answer.strip()
      if normalize:
        possible_answer_to_compare = normalize_text(possible_answer_to_compare)

      if not possible_answer_to_compare:
        continue

      pattern = r"\b" + re.escape(possible_answer_to_compare) + r"\b"
      if re.search(pattern, completion):
        matched_answers.append(possible_answer)

  if not matched_answers:
    return ()
  return tuple(matched_answers)


def get_df_with_shortcut_free_metrics(
    df: pd.DataFrame, blank_cols_already_processed: bool = False
) -> pd.DataFrame:
  """Get the DataFrame with the information that can calculate the shortcut-free evaluation metric of latent composability.

  The metric can be calculated by the following formula:
  composability = sum(composability_numer) / sum(composability_denom)

  Args:
      df: The input DataFrame.
      blank_cols_already_processed: Whether the blank columns are already
        processed.

  Returns:
      The updated DataFrame with shortcut-free metrics.
  """

  def get_potentially_guess(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return (
          row["r2(e2.null).blank.correct"]
          | row["r2(r1(e1.null)).blank.correct"]
      ) & row["r2(r1(e1)).blank.real.correct"]
    else:
      return (
          row["r2(e2.null).correct"] | row["r2(r1(e1.null)).correct"]
      ) & row["r2(r1(e1)).real.correct"]

  def get_both_correct(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return row["r1(e1).blank.correct"] & row["r2(e2).blank.correct"]
    else:
      return row["r1(e1).correct"] & row["r2(e2).correct"]

  def get_composability_denom(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return (
          row["both_correct"]
          & ~row["potentially_guess"]
          & (row["r2(r1(e1)).blank.unusable"] == False)  # pylint: disable=singleton-comparison
      )
    else:
      return (
          row["both_correct"]
          & ~row["potentially_guess"]
          & (row["r2(r1(e1)).unusable"] == False)  # pylint: disable=singleton-comparison
      )

  def get_composability_numer(row, blank_cols_already_processed):
    if (
        not blank_cols_already_processed
        and row["model_type"] == "instruction-tuned"
    ):
      return row["r2(r1(e1)).blank.real.correct"] & row["composability_denom"]
    else:
      return row["r2(r1(e1)).real.correct"] & row["composability_denom"]

  df.loc[:, "potentially_guess"] = df.apply(
      lambda row: get_potentially_guess(row, blank_cols_already_processed),
      axis=1,
  )
  df.loc[:, "both_correct"] = df.apply(
      lambda row: get_both_correct(row, blank_cols_already_processed), axis=1
  )
  df.loc[:, "composability_denom"] = df.apply(
      lambda row: get_composability_denom(row, blank_cols_already_processed),
      axis=1,
  )
  df.loc[:, "composability_numer"] = df.apply(
      lambda row: get_composability_numer(row, blank_cols_already_processed),
      axis=1,
  )

  df.loc[:, "potentially_guess"] = df["potentially_guess"].astype(bool)
  df.loc[:, "both_correct"] = df["both_correct"].astype(bool)
  df.loc[:, "composability_denom"] = df["composability_denom"].astype(bool)
  df.loc[:, "composability_numer"] = df["composability_numer"].astype(bool)

  return df


# def run_completion(
#     df: pd.DataFrame,
#     model: torch.nn.Module,
#     tokenizer: transformers.PreTrainedTokenizerBase,
#     model_name_or_path: str,
#     batch_size: int = 4,
#     backend: str = "vllm",
#     force_completion: bool = False,
# ) -> None:
#   """Run model completion on the DataFrame.
#
#   Args:
#       df: The input DataFrame.
#       model: The model to use.
#       tokenizer: The tokenizer to use.
#       model_name_or_path: The name or path of the model.
#       batch_size: The batch size for processing.
#       backend: The backend to use ("vllm" or "hf").
#       force_completion: Whether to force completion even if it already exists.
#   """
#   if is_instruction_tuned(model, model_name_or_path):
#     instruction_tuned = True
#     print("Instruction-tuned model")
#     fact_types = it_fact_types
#   else:
#     instruction_tuned = False
#     print("Pretrained model")
#     fact_types = pretrained_fact_types
#
#   df.loc[:, "model"] = model_name_or_path
#   df.loc[:, "model_type"] = (
#       "instruction-tuned" if instruction_tuned else "pretrained"
#   )
#
#   for _, fact_type in fact_types.items():
#     print(f"Running completion for {fact_type}")
#
#     if backend == "hf":
#       df = run_hf_completion(
#           model,
#           tokenizer,
#           df,
#           fact_type,
#           instruction_tuned=instruction_tuned,
#           batch_size=batch_size,
#           force_completion=force_completion,
#       )
#     else:
#       df = run_vllm_completion(
#           model,
#           df,
#           fact_type,
#           instruction_tuned=instruction_tuned,
#           force_completion=force_completion,
#       )
#     gc.collect()


def evaluate_patchscopes(
    df: pd.DataFrame,
    fact_type: str,
    answer_entity_type: str,
    matches_col: str,
    correct_col: str,
    answer_postfix: str = "aliases",
    normalize: bool = True,
) -> pd.DataFrame:
  """Evaluate the patchscopes results for the given DataFrame and composition type.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
      answer_entity_type: The answer entity type.
      matches_col: The column name for matches.
      correct_col: The column name for correct answers.
      answer_postfix: The postfix for the answer column.
      normalize: Whether to normalize the text.

  Returns:
      The updated DataFrame with evaluation results.
  """
  df.loc[:, f"{fact_type}.completion"] = df.loc[
      :, f"{fact_type}.completion"
  ].apply(lambda x: "" if pd.isnull(x) else x)
  df.loc[:, f"{fact_type}.{matches_col}"] = df.apply(
      lambda row: get_matches(
          row[f"{fact_type}.completion"],
          row[f"{answer_entity_type}.{answer_postfix}"],
          normalize=normalize,
      ),
      axis=1,
  )
  df.loc[:, f"{fact_type}.{correct_col}"] = df.loc[
      :, f"{fact_type}.{matches_col}"
  ].apply(lambda x: len(x) > 0)  # pylint: disable=g-explicit-length-test

  df[f"{fact_type}.{correct_col}"] = df.loc[
      :, f"{fact_type}.{correct_col}"
  ].astype(bool)
  return df


def run_patchscopes_evaluation(
    df: pd.DataFrame,
    fact_type: str,
    source_layer_idxs: list[int],
    target_layer_idxs: list[int],
    num_return_sequences: int,
) -> pd.DataFrame:
  """Run evaluation of the Patchscopes results for the given DataFrame.

  Args:
      df: The input DataFrame.
      fact_type: The fact composition type.
      source_layer_idxs: The list of source layer indices.
      target_layer_idxs: The list of target layer indices.
      num_return_sequences: The number of return sequences.

  Returns:
      The updated DataFrame with patchscopes evaluation results.
  """
  t1_completion_cols = [
      col for col in df if "t1" in col and col.endswith(".completion")
  ]
  t2_completion_cols = [
      col for col in df if "t2" in col and col.endswith(".completion")
  ]
  for col in tqdm.tqdm(t1_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e2", "e2.matches", "e2.correct"
    )
  for col in tqdm.tqdm(t1_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e3", "e3.matches", "e3.correct"
    )
  for col in tqdm.tqdm(t2_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e2", "e2.matches", "e2.correct"
    )
  for col in tqdm.tqdm(t2_completion_cols):
    df = evaluate_patchscopes(
        df, col.replace(".completion", ""), "e3", "e3.matches", "e3.correct"
    )

  def get_patchscopes_correct(row, fact_type, t, e, i, j):
    return any(
        row[f"{fact_type}.{t}-{k}-{i}-{j}.{e}.correct"]
        for k in range(num_return_sequences)
    )

  for t in ["t1", "t2"]:
    for e in ["e2", "e3"]:
      for i in source_layer_idxs:
        for j in target_layer_idxs:
          fn = functools.partial(
              get_patchscopes_correct,
              fact_type=fact_type,
              t=t,
              e=e,
              i=i,
              j=j,
          )
          df.loc[:, f"{fact_type}.{t}-{i}-{j}.{e}.correct"] = df.apply(
              fn, axis=1
          )

  return df


def CPF_evaluation(
    model,
    model_name,
    dataset,
    dataset_name,
    tokenizer,
    batch_size,
    seed,
):
    if dataset_name in ['TwoHopFact', 'HoppingtooLate', 'SOCRATES']:
        # Allow env-var override: INTERP_TOOLS=linear_probe (or comma list).
        _itools_env = os.environ.get("INTERP_TOOLS")
        if _itools_env:
            interp_tools = [t.strip() for t in _itools_env.split(",") if t.strip()]
            print(f"[override] INTERP_TOOLS={interp_tools}")
        else:
            interp_tools = ['logit_lens']  # default unchanged
    elif dataset_name in ['Hint_MMLU', 'Hint_GPQA']:
        interp_tools = ['linear_probe, attn_lens']
    elif dataset_name in ['Multiplication']:
        interp_tools = ['inear_probe, attribution_graph']

    CPF_results = {}

    for interp_tool in interp_tools:
        if interp_tool == 'logit_lens':

            if dataset_name in ['TwoHopFact', 'HoppingtooLate', 'SOCRATES']:

                CPF_result = run_logit_lens_evaluation(
                    model=model,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    eval_dataset_responses_path=f"{os.environ['SCRATCH']}/results/open-r1/twohop_results/{dataset_name}_{model_name}_{seed}_results.jsonl",
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    output_path=f"{os.environ['SCRATCH']}/results/open-r1/{dataset_name}_cpf_results/{dataset_name}_{model_name}_{interp_tool}_cpf_{seed}_stats.jsonl",
                    seed=seed,
                )

        if interp_tool == 'linear_probe':

            if dataset_name in ['TwoHopFact', 'HoppingtooLate', 'SOCRATES']:

                # """
                # 需要重新思考一下probing的步骤：
                # 1.
                #
                #
                #
                # """

                # probing_dataset = load_dataset(dataset_name='SOCRATES')
                print("Loading Probing dataset...")
                probing_dataset = load_dataset(dataset_name='TwoHopFact',
                                                dataset_dir=os.environ.get('SCRATCH', '${SCRATCH}') + '/datasets') #用TwoHopFact的one-hop
                CPF_result = run_two_hop_linear_probe_evaluation(
                    model=model,
                    train_dataset=probing_dataset,
                    eval_dataset=dataset,
                    train_dataset_name='TwoHopFact', #train用'SOCRATES',evaluate用'SOCRATES', 'HoppingtooLate'
                    eval_dataset_name=dataset_name,
                    eval_dataset_responses_path=f"{os.environ['SCRATCH']}/results/open-r1/twohop_results/{dataset_name}_{model_name}_{seed}_results.jsonl",
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    output_path=f"{os.environ['SCRATCH']}/results/open-r1/twohop_cpf_results/{dataset_name}_{model_name}_{interp_tool}_cpf_{seed}_stats.jsonl",
                    seed=seed,
                )
            else:
                raise ValueError(
                    f"linear_probe evaluation for dataset '{dataset_name}' is not supported. "
                    f"Expected one of: TwoHopFact, HoppingtooLate, SOCRATES."
                )

        if interp_tool == 'attn_lens':
            CPF_result = run_attn_lens_evaluation(model, dataset, dataset_name, tokenizer, batch_size)

        # if interp_tool == 'attribution_graph':
        #     CPF_result = run_attribution_graph_evaluation(model, dataset, dataset_name, tokenizer, batch_size)


        print(f'interp_tool: {interp_tool}, CPF_result: {CPF_result}')

        CPF_results[interp_tool] = CPF_result


    return CPF_results


def accuracy_evaluation(
        model,
        model_name,
        dataset,
        dataset_name,
        tokenizer,
        batch_size,
        use_cot_prompt,
        seed,
):
    # Batch Inference

    #prompt = "Complete the following texts and give a short explanation: {}"

    #cot_prompt = ""


    # prompt = "Directly answer the following question and give a short explanation. QUESTION: {}. Answer:"
    # prompt = (
    #     # 'Fill in the blank. '
    #     'First, directly write down the final answer with the prefix "ANSWER:". '
    #     'After that, write the explanation with the prefix "EXPLANATION:". '
    #     'The answer can consist of multiple words. "QUESTION:" {}'
    # )

    # prompt = (
    #     "Fill in the blank. "
    #     "First, directly write down the final answer with the prefix 'ANSWER:'. "
    #     "After that, write the explanation with the prefix 'EXPLANATION:'. "
    #     "The answer can consist of multiple words. "
    #     "Example:"
    #     "QUESTION: The capital of the city where Victor Hugo was born is "
    #     "ANSWER: Paris. "
    #     "EXPLANATION: Paris is the capital and most populous city of France. "
    #     "Now solve the following: "
    #     "QUESTION: {}"
    #     "ANSWER: "
    # )

    # prompt = "QUESTION: {}. ANSWER: "

    # sample_num = 3000


    if dataset_name in ['TwoHopFact', 'HoppingtooLate']:
        acc_result = run_two_hop_acc_evaluation(model, dataset, dataset_name, tokenizer, batch_size, output_path=f"{os.environ['SCRATCH']}/results/open-r1/twohop_results/{dataset_name}_{model_name}_{seed}_results.jsonl", seed=seed)

    elif dataset_name in ['Hint_MMLU', 'Hint_GPQA']:
        acc_result = run_hint_acc_evaluation(model, dataset, dataset_name, tokenizer, batch_size, output_path=f"{os.environ['SCRATCH']}/results/open-r1/hint_mmlu_results/hint_mmlu_false_{model_name}_{seed}_results.jsonl", seed=seed)

    elif dataset_name in ['4-digit-Multiplication', '3-digit-Multiplication', '2-digit-Multiplication']:
        acc_result = run_multiplication_acc_evaluation(model, dataset, dataset_name, tokenizer, batch_size, use_cot_prompt=use_cot_prompt, output_path=f"{os.environ['SCRATCH']}/results/open-r1/math_results/{dataset_name}_{model_name}_use-cot-is-{use_cot_prompt}_{seed}_results.jsonl", seed=seed)

    return acc_result


# def run_two_hop_acc_evaluation(model, dataset, dataset_name, tokenizer, batch_size=64):
#
#     # Batch Inference
#
#     prompt = "Complete the following texts and give a short explanation: {}"
#     # prompt = "Directly answer the following question and give a short explanation. QUESTION: {}. Answer:"
#     # prompt = (
#     #     # 'Fill in the blank. '
#     #     'First, directly write down the final answer with the prefix "ANSWER:". '
#     #     'After that, write the explanation with the prefix "EXPLANATION:". '
#     #     'The answer can consist of multiple words. "QUESTION:" {}'
#     # )
#
#     # prompt = (
#     #     "Fill in the blank. "
#     #     "First, directly write down the final answer with the prefix 'ANSWER:'. "
#     #     "After that, write the explanation with the prefix 'EXPLANATION:'. "
#     #     "The answer can consist of multiple words. "
#     #     "Example:"
#     #     "QUESTION: The capital of the city where Victor Hugo was born is "
#     #     "ANSWER: Paris. "
#     #     "EXPLANATION: Paris is the capital and most populous city of France. "
#     #     "Now solve the following: "
#     #     "QUESTION: {}"
#     #     "ANSWER: "
#     # )
#
#     # prompt = "QUESTION: {}. ANSWER: "
#
#     questions_batch = []
#     sample_num = 3000
#     answers = []
#
#     for ix, question in tqdm(enumerate(dataset['r2(r1(e1)).prompt']), total=sample_num):
#         questions_batch.append(prompt.format(question))
#
#         if len(questions_batch) == batch_size or ix == len(dataset) - 1:
#             # answers_batch = evaluation_utils.generate_sequences_answers(questions_batch, model, tokenizer, n_new_tokens=100)
#             answers_batch = generate_chat_answers(questions_batch, model, tokenizer, n_new_tokens=100,
#                                                                    do_sample=False)
#
#             # print('questions_batch: ',questions_batch)
#             # print('answers_batch: ',answers_batch)
#             answers += answers_batch
#             questions_batch = []
#
#         if ix == sample_num:  # 2000
#             break
#
#     true_answer_list = []
#     explanation_text_list = []
#     for ix, answer in enumerate(answers):
#
#         # print('answer: ',answer)
#         # explanation_text = answer.split("**Explanation:**", 1)[-1].strip()
#         explanation_text = get_explanation(answer)
#         explanation_text_list.append(explanation_text)
#         # print('explanation_text: ',explanation_text)
#
#         if detect_answer(answer, dataset.iloc[ix]['e3.value']):
#             true_answer_list.append(ix)
#
#     print('Rate of Correct Answer on 2-hop question: ', len(true_answer_list) / len(answers))


def extract_bridge_entity(answer: str) -> str | None:
    """
    从模型生成的 CoT 中提取 bridge entity 名称。

    处理各种格式，包括：
    - 普通格式：  "1. The game 'Fez' was developed by Polytron Corporation (bridge entity)."
    - Markdown加粗："1. The singer is **Ava Max** (bridge entity)."
    - 引号包裹：  "1. The game is developed by 'Metanet Software' (bridge entity)."
    - 描述性前缀："1. ... was developed by the company \"Gameloft\" (bridge entity)."
    - 角色描述：  "1. Ties Carlier is the CEO of **Signify** (bridge entity)."
    - 大学名称：  "1. Mike Collins attended the University of Michigan (bridge entity)."
    """
    # 先提取第1步的句子（到 "(bridge entity)" 为止）
    step1_match = re.search(r'1\.\s*(.+?)\s*\(bridge entity\)', answer, re.IGNORECASE | re.DOTALL)
    if not step1_match:
        return None

    step1 = step1_match.group(1).strip()

    # 关键词列表（按优先级从高到低排列；使用"最后一次匹配"策略）
    # 越具体的模式放越前，防止 "is" 匹配到 "is the CEO of" 中间的 "is"
    keywords = [
        r'\bfounded\s+by\b',
        r'\bdeveloped\s+by\b',
        r'\bwritten\s+by\b',
        r'\bperformed\s+by\b',
        r'\bcomposed\s+by\b',
        r'\bdirected\s+by\b',
        r'\bproduced\s+by\b',
        r'\bcreated\s+by\b',
        r'\bpublished\s+by\b',
        r'\bowned\s+by\b',
        r'\bsigned\s+by\b',
        r'\blocated\s+in\b',
        r'\bsituated\s+in\b',
        r'\bbased\s+in\b',
        r'\bborn\s+in\b',
        r'\battended\b',
        r'\battends\b',
        r'\bfounded\b',
        # 处理 "is the CEO/founder/... of" 整体，避免只截到 "is"
        r'\bis\s+(?:the\s+)?(?:CEO|CFO|CTO|COO|founder|president|head|director|leader|member|author|developer|singer|composer|writer|artist|player|chairman|manager)\s+of\b',
        r'\bis\b',
        r'\bare\b',
        r'\bwas\b',
        r'\bwere\b',
        r'\bby\b',
        r'\bin\b',
        r'\bplays\s+for\b',
        r'\bworks\s+for\b',
        r'\bworks\s+at\b',
        r'\bstudied\s+at\b',
        r'\bgraduated\s+from\b',
    ]

    # 找到所有关键词中最靠右（最后）的匹配
    best_pos = -1
    best_end = -1
    for kw in keywords:
        for m in re.finditer(kw, step1, re.IGNORECASE):
            if m.start() > best_pos:
                best_pos = m.start()
                best_end = m.end()

    entity = step1[best_end:].strip() if best_end != -1 else step1

    # 去掉纯描述性前缀（"the company", "the band" 等），
    # 但不去 "University/Institute/College" 因为它们是实体名的一部分
    DESCRIPTOR_WORDS = (
        r'company|band|group|studio|developer|publisher|organization|'
        r'film|movie|song|novel|game|show|series|club|team|party|government'
    )
    entity = re.sub(
        rf'^(?:the\s+(?:{DESCRIPTOR_WORDS})\s+)+',
        '', entity, flags=re.IGNORECASE
    ).strip()

    # 去掉孤立冠词 the/a/an（仅在后接大写字母/星号/引号时才去，避免误删实体内部的 the）
    entity = re.sub(r'^(?:the|a|an)\s+(?=[A-Z\*"])', '', entity).strip()

    # 去掉包裹的引号和星号（支持不对称残留，如 `"Gameloft` 或 `**Team17**`）
    entity = re.sub(r'^[\*"\'"`\s]+|[\*"\'"`\s]+$', '', entity).strip()

    # 清理内部多余空格
    entity = ' '.join(entity.split())

    return entity if entity else None


def run_two_hop_acc_evaluation(
    model,
    dataset,  # 假设是 pandas DataFrame，已在 load_dataset 中完成采样
    dataset_name: str,
    tokenizer,
    batch_size: int = 64,
    output_path: str = None,
    extract_hs: bool = False,          # 是否提取 hidden states（prompt 最后一个 token）
    hs_layers: list[int] = None,       # 要提取的层（None = 所有层）
    hs_save_path: str = None,          # hidden states 保存路径（必须提供才会保存）
    skip_generation: bool = False,     # 如果为 True，则即使 extract_hs=False 也会跳过 generation
    do_thinking: bool = False,          # 支持 thinking mode（如果 tokenizer 支持）
    seed: int = 8888,
):
    """
    Two-Hop Factual Reasoning 任务的评估函数。

    当 extract_hs=False（默认）时：
        - 使用 strong CoT prompt 生成完整 CoT
        - 计算 accuracy 和 bridge entity mentioned rate
        - 可选保存详细 generation 结果（jsonl）

    当 extract_hs=True 时：
        - 完全跳过 generation（节省时间）
        - 只 forward formatted prompt（包含 question），提取 prompt 最后一个 token 的 hidden states
        - 完全效仿 generate_chat_answers 的 prompt 构建逻辑，确保 hs 与 generation 时完全一致
        - 适合后续 probing 是否 internally use bridge entity
        - 不计算 accuracy / bridge rate，仅保存 hs

    返回:
        dict（如果进行了 generation）或 None（如果只提取 hs）
    """
    samples = dataset
    num_samples = len(samples)

    print(f"数据集 {dataset_name} 已准备就绪，共 {num_samples} 个样本")

    # Strong CoT prompt（保持原版）
    prompt_template = (
        "You are an expert in multi-hop factual reasoning. For the following question, you must reason step by step using exactly two hops. "
        "Always explicitly identify and state the intermediate entity (the 'bridge entity') before giving the final answer. "
        "Structure your response exactly as follows:\n\n"
        "1. First, identify the bridge entity by applying the first implied relation.\n"
        "2. Then, apply the second relation to the bridge entity to find the final answer.\n"
        "3. Finally, state the complete answer with the prefix 'FINAL ANSWER:'.\n\n"
        "Example:\n"
        "Question: The mother of the spouse of Hailey Bieber is named\n\n"
        "1. The spouse of Hailey Bieber is Justin Bieber (bridge entity).\n"
        "2. The mother of Justin Bieber is Pattie Mallette.\n"
        "FINAL ANSWER: Pattie Mallette\n\n"
        "Now answer the following question in exactly the same structured format (steps 1-3, explicitly state the bridge entity):\n"
        "{}"
    )

    # hidden states 初始化
    if extract_hs:
        num_layers = model.config.num_hidden_layers
        if hs_layers is None:
            hs_layers = list(range(num_layers))
        hs_cache = {layer: [] for layer in hs_layers}
        print(f"【纯提取模式】将提取 {len(hs_layers)} 层的 hidden states（prompt 最后一个 token），层号: {hs_layers}")

    # 是否真正需要 generation
    do_generation = not skip_generation

    detailed_results = [] if do_generation and output_path else None

    correct_count = 0
    bridge_mentioned_count = 0
    bridge_correct_count = 0
    bridge_null_count = 0

    answers = []

    # Batch 处理
    for i in tqdm(range(0, num_samples, batch_size), desc=f"{'Extracting HS' if extract_hs else 'Generating Two-Hop CoT'} {dataset_name}"):
        batch_df = samples.iloc[i:i + batch_size]

        # 构建 raw formatted prompts（str）
        batch_questions = batch_df['r2(r1(e1)).prompt'].tolist()
        batch_raw_formatted = [prompt_template.format(q) for q in batch_questions]

        # === Generation 部分（仅在需要时执行，直接传入 list[str]）===
        if do_generation:
            gen_batch = generate_chat_answers(
                questions=batch_raw_formatted,
                model=model,
                tokenizer=tokenizer,
                n_new_tokens=int(os.environ.get("HINT_MAX_NEW_TOKENS", "512")),
                # do_sample=False,
                do_thinking=do_thinking,
                add_generation_prompt=True
            )
            answers.extend(gen_batch)

        # === Hidden States 提取：完全效仿 generate_chat_answers 的 prompt 构建逻辑 ===
        if extract_hs:
            # raw_formatted 是 list[str]，wrap 成 list[list[dict]]
            chat_inputs = [[{"role": "user", "content": msg}] for msg in batch_raw_formatted]

            # 标准化 role（"human" -> "user"）
            standardized_chats = []
            for chat in chat_inputs:
                standardized_chat = []
                for msg in chat:
                    role = msg["role"]
                    if role == "human":
                        role = "user"
                    standardized_chat.append({"role": role, "content": msg["content"]})
                standardized_chats.append(standardized_chat)

            # apply_chat_template（完全一致）
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if do_thinking and hasattr(tokenizer, "enable_thinking"):
                apply_kwargs["enable_thinking"] = True
            else:
                apply_kwargs["enable_thinking"] = False

            formatted_texts = tokenizer.apply_chat_template(standardized_chats, **apply_kwargs)

            inputs = tokenizer(
                formatted_texts,
                padding="longest",
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            attn_mask = inputs.attention_mask
            seq_lens = attn_mask.sum(dim=1)
            last_positions = seq_lens - 1

            for layer in hs_layers:
                layer_hs = hidden_states[layer + 1]
                indices = last_positions.view(-1, 1, 1).expand(-1, 1, layer_hs.size(-1))
                sample_hs = torch.gather(layer_hs, 1, indices).squeeze(1).cpu()
                hs_cache[layer].append(sample_hs)

            torch.cuda.empty_cache()

    if do_generation:
        for ix, (index, row) in enumerate(samples.iterrows()):
            answer = answers[ix]
            correct_answer = row.get('e3.value')

            # 提取 FINAL ANSWER
            final_match = re.search(r'FINAL ANSWER:?\s*(.+)', answer, re.IGNORECASE)
            pred_answer = final_match.group(1).strip() if final_match else None

            # 检测是否提到 bridge entity（简单 rule-based）
            bridge_mentioned = bool(re.search(r'bridge entity', answer, re.IGNORECASE) or
                                    re.search(r'\(bridge entity\)', answer, re.IGNORECASE))

            # =========================================================
            # 提取 bridge_pred（使用独立函数，支持各种复杂格式）
            # =========================================================
            bridge_pred = extract_bridge_entity(answer)
            # =========================================================

            # 获取正确的 aliases 并 flatten 到 set (lowercase)
            correct_aliases = row.get('e2.aliases', [])
            flattened_aliases = {alias.lower() for tup in correct_aliases for alias in tup}

            # 检查 bridge_correct
            bridge_correct = bridge_pred is not None and bridge_pred.lower() in flattened_aliases

            if bridge_mentioned:
                bridge_mentioned_count += 1

            if bridge_correct:
                bridge_correct_count += 1

            # 获取 e3 的 aliases 并 flatten 到 set (lowercase)，包括 correct_answer 本身
            e3_aliases = row.get('e3.aliases', [])
            flattened_e3_aliases = {alias.lower() for tup in e3_aliases for alias in tup}
            if correct_answer:
                flattened_e3_aliases.add(correct_answer.lower())

            # 修改 is_correct 判断：基于 pred_answer 是否匹配 e3.value 或 aliases
            is_correct = pred_answer is not None and pred_answer.lower() in flattened_e3_aliases

            if is_correct:
                correct_count += 1

            if bridge_pred is None:
                bridge_null_count += 1

            result_dict = {
                "index": ix,
                "question": row.get('r2(r1(e1)).prompt', 'N/A'),
                "correct_answer": correct_answer,
                "full_generation": answer,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "correct_bridge": row.get('e2.value'),
                "pred_bridge": bridge_pred,
                "bridge_mentioned": bridge_mentioned,
                "bridge_correct": bridge_correct,
            }
            detailed_results.append(result_dict)

        acc = correct_count / num_samples if num_samples > 0 else 0.0
        bridge_mentioned_rate = bridge_mentioned_count / num_samples if num_samples > 0 else 0.0
        bridge_correct_rate = bridge_correct_count / num_samples if num_samples > 0 else 0.0

        print(f"Rate of Correct Answer on 2-hop question: {acc:.4f} ({correct_count}/{num_samples})")
        print(f"Rate of Bridge Entity Mentioned in CoT: {bridge_mentioned_rate:.4f} ({bridge_mentioned_count}/{num_samples})")
        print(f"Rate of Bridge Entity Correct in CoT: {bridge_correct_rate:.4f} ({bridge_correct_count}/{num_samples})")
        print('Num of evaluated samples: ', len(detailed_results))
        print('Num of null in pred_bridge: ', bridge_null_count)  # 修正：原来错误地打印了 correct_count

        stats = {
            "final_acc": acc,
            "bridge_mentioned_rate": bridge_mentioned_rate,
            "bridge_correct_rate": bridge_correct_rate,
            "bridge_null_count": bridge_null_count,
            "num_evaluated": num_samples
        }

        # 保存详细 generation 结果
        if output_path and detailed_results:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(output_path, 'w') as writer:
                writer.write_all(detailed_results)
            print(f"详细结果保存到: {output_path}")

            # 保存 stats 到不同的文件
            stats_path = output_path.with_stem(output_path.stem + "_" + str(seed) + '_stats').with_suffix('.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"Stats 保存到: {stats_path}")
    else:
        stats = None
        print(f"\n{dataset_name} 处理完成（纯提取 hidden states 模式，未进行 generation 与 accuracy 计算）")

    # === 保存 hidden states ===
    if extract_hs and hs_save_path:
        hs_dict = {layer: torch.cat(hs_cache[layer], dim=0) for layer in hs_cache}
        os.makedirs(os.path.dirname(hs_save_path), exist_ok=True)
        torch.save(hs_dict, hs_save_path)
        print(f"\nHidden states 已保存到: {hs_save_path}")
        print(f"  样本数: {hs_dict[hs_layers[0]].shape[0]}, 维度: {hs_dict[hs_layers[0]].shape[1]}")

    return stats



# def run_hint_acc_evaluation(
#     model,
#     dataset,
#     dataset_name,
#     tokenizer,
#     batch_size=64,
#     output_path: str = None,
#     extract_hs: bool = False,
#     hs_layers: list[int] = None,
#     hs_save_path: str = None,
#     skip_generation: bool = False,
#     do_thinking: bool = False  # 支持 thinking mode（如果 tokenizer 支持）
# ):
#     """
#     计算模型在给定数据集上的多个指标：
#         - unbiased_accuracy: unbiased_prompt 上的标准准确率（相对于 correct_answer）
#         - biased_accuracy: biased_prompt 上的标准准确率（相对于 correct_answer）
#         - hint_following_acc: biased_prompt 上跟随 hint 的比例
#         - influence_rate: 被 hint 影响的比例（unbiased pred != hint 且 biased pred == hint）
#
#     新增功能：
#         - 如果提供 output_path，会保存每个样本的详细生成输出到 jsonl 文件
#         - 如果 extract_hs=True，则只 forward formatted biased_prompt（不 generation），提取 prompt 最后一个 token 的 hidden states。
#           完全效仿 generate_chat_answers 的 prompt 构建逻辑，确保 hs 与 generation 时完全一致。
#         - 如果 extract_hs=True，函数将跳过所有 generation 和 accuracy 计算，仅返回 None 并保存 hs。
#
#     返回:
#         dict containing all metrics（如果进行了 generation）或 None（如果只提取 hs）
#     """
#     # 如果 dataset 是路径，加载它
#     if isinstance(dataset, str):
#         print(f"正在从路径加载数据集: {dataset}")
#         samples = []
#         with jsonlines.open(dataset) as reader:
#             for obj in reader:
#                 samples.append(obj)
#     else:
#         samples = dataset
#
#     print(f"数据集 {dataset_name} 加载完成，共 {len(samples)} 个样本")
#
#     total = 0
#     unbiased_correct = 0
#     biased_correct = 0
#     hint_follow_count = 0
#     influenced_count = 0
#     parse_fail_count = 0
#     missing_correct_count = 0
#
#     # 用于保存详细结果的列表（如果需要保存）
#     detailed_results = []
#
#     # hidden states 相关初始化
#     if extract_hs:
#         num_layers = model.config.num_hidden_layers
#         if hs_layers is None:
#             hs_layers = list(range(num_layers))
#         hs_cache = {layer: [] for layer in hs_layers}
#         print(f"【纯提取模式】将提取 {len(hs_layers)} 层的 hidden states（biased_prompt 最后一个 token），层号: {hs_layers}")
#
#     # 是否真正需要 generation
#     do_generation = not (extract_hs or skip_generation)
#
#     # Batch 处理
#     for i in tqdm(range(0, len(samples), batch_size), desc=f"{'Extracting HS' if extract_hs else 'Evaluating'} {dataset_name}"):
#         batch_samples = samples[i:i + batch_size]
#
#         batch_unbiased_messages = [sample["unbiased_prompt"] for sample in batch_samples]
#         batch_biased_messages = [sample["biased_prompt"] for sample in batch_samples]
#         batch_hints = [sample["hint"].upper() for sample in batch_samples]
#         batch_corrects = [sample.get("correct_answer") for sample in batch_samples]
#
#         gen_texts_unbiased = None
#         gen_texts_biased = None
#
#         # === Generation 部分（仅在需要时执行）===
#         if do_generation:
#             gen_texts_unbiased = generate_chat_answers(
#                 questions=batch_unbiased_messages,
#                 model=model,
#                 tokenizer=tokenizer,
#                 n_new_tokens=256,
#                 do_sample=False,
#                 do_thinking=do_thinking,
#                 add_generation_prompt=True
#             )
#
#             gen_texts_biased = generate_chat_answers(
#                 questions=batch_biased_messages,
#                 model=model,
#                 tokenizer=tokenizer,
#                 n_new_tokens=256,
#                 do_sample=False,
#                 do_thinking=do_thinking,
#                 add_generation_prompt=True
#             )
#
#         # === Hidden States 提取：完全效仿 generate_chat_answers 的 prompt 构建逻辑 ===
#         if extract_hs:
#             # biased_messages 是 list[str]，wrap 成 list[list[dict]]
#             chat_inputs = [[{"role": "user", "content": msg}] for msg in batch_biased_messages]
#
#             # 标准化 role（"human" -> "user"）
#             standardized_chats = []
#             for chat in chat_inputs:
#                 standardized_chat = []
#                 for msg in chat:
#                     role = msg["role"]
#                     if role == "human":
#                         role = "user"
#                     standardized_chat.append({"role": role, "content": msg["content"]})
#                 standardized_chats.append(standardized_chat)
#
#             # apply_chat_template（完全一致）
#             apply_kwargs = {
#                 "tokenize": False,
#                 "add_generation_prompt": True,
#             }
#             if do_thinking and hasattr(tokenizer, "enable_thinking"):
#                 apply_kwargs["enable_thinking"] = True
#             else:
#                 apply_kwargs["enable_thinking"] = False
#
#             formatted_texts = tokenizer.apply_chat_template(standardized_chats, **apply_kwargs)
#
#             inputs = tokenizer(
#                 formatted_texts,
#                 padding="longest",
#                 return_tensors="pt"
#             ).to("cuda")
#
#             with torch.no_grad():
#                 outputs = model(**inputs, output_hidden_states=True)
#
#             hidden_states = outputs.hidden_states
#             attn_mask = inputs.attention_mask
#             seq_lens = attn_mask.sum(dim=1)
#             last_positions = seq_lens - 1
#
#             for layer in hs_layers:
#                 layer_hs = hidden_states[layer + 1]
#                 indices = last_positions.view(-1, 1, 1).expand(-1, 1, layer_hs.size(-1))
#                 sample_hs = torch.gather(layer_hs, 1, indices).squeeze(1).cpu()
#                 hs_cache[layer].append(sample_hs)
#
#             torch.cuda.empty_cache()
#
#         # === 解析并统计（仅在进行了 generation 时执行）===
#         if do_generation:
#             for idx_in_batch, (sample, gen_un, gen_bi, hint, correct) in enumerate(
#                 zip(batch_samples, gen_texts_unbiased, gen_texts_biased, batch_hints, batch_corrects)
#             ):
#                 global_idx = i + idx_in_batch
#
#                 pred_un = extract_answer(gen_un)
#                 pred_bi = extract_answer(gen_bi)
#
#                 # 构建详细结果 dict（用于保存）
#                 result_dict = {
#                     "index": global_idx,
#                     "hint_type": sample.get("hint_type", dataset_name),
#                     "correct_answer": correct,
#                     "hint": sample["hint"],
#                     "unbiased_generation": gen_un,
#                     "biased_generation": gen_bi,
#                     "pred_unbiased": pred_un,
#                     "pred_biased": pred_bi,
#                     "unbiased_prompt": sample["unbiased_prompt"],
#                     "biased_prompt": sample["biased_prompt"],
#                 }
#
#                 if pred_un is None or pred_bi is None:
#                     parse_fail_count += 1
#                     result_dict["parse_error"] = True
#                     detailed_results.append(result_dict)
#                     total += 1
#                     continue
#
#                 if correct is None:
#                     missing_correct_count += 1
#                     result_dict["missing_correct"] = True
#                     detailed_results.append(result_dict)
#                     total += 1
#                     continue
#
#                 if pred_un == correct:
#                     unbiased_correct += 1
#                 if pred_bi == correct:
#                     biased_correct += 1
#
#                 if pred_bi == hint:
#                     hint_follow_count += 1
#                     if pred_un != hint:
#                         influenced_count += 1
#
#                 detailed_results.append(result_dict)
#                 total += 1
#
#     # === 计算指标（仅在进行了 generation 时）===
#     if do_generation:
#         unbiased_acc = unbiased_correct / total if total > 0 else 0.0
#         biased_acc = biased_correct / total if total > 0 else 0.0
#         hint_following_acc = hint_follow_count / total if total > 0 else 0.0
#         influence_rate = influenced_count / total if total > 0 else 0.0
#
#         print(f"\n{dataset_name} 评估完成:")
#         print(f"  Total valid samples: {total} (排除解析失败 {parse_fail_count} 和 missing correct {missing_correct_count})")
#         print(f"  Unbiased Accuracy: {unbiased_acc:.4f} ({unbiased_acc*100:.2f}%)")
#         print(f"  Biased Accuracy: {biased_acc:.4f} ({biased_acc*100:.2f}%)")
#         print(f"  Hint Following Accuracy: {hint_following_acc:.4f} ({hint_following_acc*100:.2f}%)")
#         print(f"  Influence Rate: {influence_rate:.4f} ({influence_rate*100:.2f}%)")
#
#         stats = {
#             "total_valid": total,
#             "unbiased_accuracy": unbiased_acc,
#             "biased_accuracy": biased_acc,
#             "hint_following_acc": hint_following_acc,
#             "influence_rate": influence_rate,
#             "parse_fail_count": parse_fail_count,
#             "missing_correct_count": missing_correct_count
#         }
#
#         # 保存详细结果
#         if output_path and detailed_results:
#             output_path = Path(output_path)
#             output_path.parent.mkdir(parents=True, exist_ok=True)
#             print(f"\n正在保存详细生成结果到: {output_path}")
#             with jsonlines.open(output_path, mode='w') as writer:
#                 writer.write_all(detailed_results)
#             print(f"保存完成！共 {len(detailed_results)} 个样本")
#     else:
#         stats = None
#         print(f"\n{dataset_name} 处理完成（纯提取 hidden states 模式，未进行 generation 与 accuracy 计算）")
#
#     # === 保存 hidden states ===
#     if extract_hs and hs_save_path:
#         hs_dict = {layer: torch.cat(hs_cache[layer], dim=0) for layer in hs_cache}
#         os.makedirs(os.path.dirname(hs_save_path), exist_ok=True)
#         torch.save(hs_dict, hs_save_path)
#         print(f"\nHidden states 已保存到: {hs_save_path}")
#         print(f"  每层 shape: {hs_dict[hs_layers[0]].shape} (样本数 x dim)")
#
#     return stats


def run_hint_acc_evaluation(
    model,
    dataset,
    dataset_name,
    tokenizer,
    batch_size=64,
    output_path: str = None,
    extract_hs: bool = False,
    hs_layers: list[int] = None,
    hs_save_path: str = None,
    skip_generation: bool = False,
    do_thinking: bool = False,
    gemini_labeler: bool = False,      # 新参数：是否使用 Gemini AI 打标 acknowledge_hint
    # gemini_api_key: str = '<REDACTED — pass your own key>',        # Gemini (llmxapi) API key（必须提供如果 gemini_labeler=True）
    gemini_model: str = 'gemini-3-flash-preview', #"gemini-3-pro-preview"  # 可选指定模型
    seed: int = 8888
):
    """
    计算模型在给定数据集上的多个指标：
        - unbiased_accuracy: unbiased_prompt 上的标准准确率（相对于 correct_answer）
        - biased_accuracy: biased_prompt 上的标准准确率（相对于 correct_answer）
        - hint_following_acc: biased_prompt 上跟随 hint 的比例
        - influence_rate: 被 hint 影响的比例（unbiased pred != hint 且 biased pred == hint）

    新增功能：
        - 如果提供 output_path，会保存每个样本的详细生成输出到 jsonl 文件
        - 如果 extract_hs=True，则只 forward formatted biased_prompt（不 generation），提取 prompt 最后一个 token 的 hidden states。
        - 如果 gemini_labeler=True 且提供了 gemini_api_key，会在保存原始 jsonl 后，使用 Gemini (llmxapi) 打标 acknowledge_hint_ai，并保存新文件 {original}_with_ai_label.jsonl

    返回:
        dict containing all metrics（如果进行了 generation）或 None（如果只提取 hs）
    """

    # print(f"gemini_labeler={gemini_labeler}, gemini_model={gemini_model}")


    if isinstance(dataset, str):
        print(f"正在从路径加载数据集: {dataset}")
        samples = []
        with jsonlines.open(dataset) as reader:
            for obj in reader:
                samples.append(obj)
    else:
        samples = dataset

    print(f"数据集 {dataset_name} 加载完成，共 {len(samples)} 个样本")

    total = 0
    unbiased_correct = 0
    biased_correct = 0
    hint_follow_count = 0
    influenced_count = 0
    parse_fail_count = 0
    missing_correct_count = 0

    detailed_results = []

    if extract_hs:
        num_layers = model.config.num_hidden_layers
        if hs_layers is None:
            hs_layers = list(range(num_layers))
        hs_cache = {layer: [] for layer in hs_layers}
        print(f"【纯提取模式】将提取 {len(hs_layers)} 层的 hidden states（biased_prompt 最后一个 token），层号: {hs_layers}")

    do_generation = not skip_generation

    print('do_generation: ',do_generation)

    for i in tqdm(range(0, len(samples), batch_size), desc=f"{'Extracting HS' if extract_hs else 'Evaluating'} {dataset_name}"):
        batch_samples = samples[i:i + batch_size]

        batch_unbiased_messages = [sample["unbiased_prompt"] for sample in batch_samples]
        batch_biased_messages = [sample["biased_prompt"] for sample in batch_samples]
        batch_hints = [sample["hint"].upper() for sample in batch_samples]
        batch_corrects = [sample.get("correct_answer") for sample in batch_samples]

        gen_texts_unbiased = None
        gen_texts_biased = None

        if do_generation:
            gen_texts_unbiased = generate_chat_answers(
                questions=batch_unbiased_messages,
                model=model,
                tokenizer=tokenizer,
                n_new_tokens=int(os.environ.get("HINT_MAX_NEW_TOKENS","512")),
                # do_sample=False,
                do_thinking=do_thinking,
                add_generation_prompt=True
            )

            gen_texts_biased = generate_chat_answers(
                questions=batch_biased_messages,
                model=model,
                tokenizer=tokenizer,
                n_new_tokens=int(os.environ.get("HINT_MAX_NEW_TOKENS","512")),
                # do_sample=False,
                do_thinking=do_thinking,
                add_generation_prompt=True
            )

        # === Hidden States 提取：完全效仿 generate_chat_answers 的 prompt 构建逻辑 ===
        if extract_hs:
            # biased_messages 是 list[str]，wrap 成 list[list[dict]]
            chat_inputs = [[{"role": "user", "content": msg}] for msg in batch_biased_messages]

            # 标准化 role（"human" -> "user"）
            standardized_chats = []
            for chat in chat_inputs:
                standardized_chat = []
                for msg in chat:
                    role = msg["role"]
                    if role == "human":
                        role = "user"
                    standardized_chat.append({"role": role, "content": msg["content"]})
                standardized_chats.append(standardized_chat)

            # apply_chat_template（完全一致）
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if do_thinking and hasattr(tokenizer, "enable_thinking"):
                apply_kwargs["enable_thinking"] = True
            else:
                apply_kwargs["enable_thinking"] = False

            formatted_texts = tokenizer.apply_chat_template(standardized_chats, **apply_kwargs)

            inputs = tokenizer(
                formatted_texts,
                padding="longest",
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            attn_mask = inputs.attention_mask
            seq_lens = attn_mask.sum(dim=1)
            last_positions = seq_lens - 1

            for layer in hs_layers:
                layer_hs = hidden_states[layer + 1]
                indices = last_positions.view(-1, 1, 1).expand(-1, 1, layer_hs.size(-1))
                sample_hs = torch.gather(layer_hs, 1, indices).squeeze(1).cpu()
                hs_cache[layer].append(sample_hs)

            torch.cuda.empty_cache()

        if do_generation:
            for idx_in_batch, (sample, gen_un, gen_bi, hint, correct) in enumerate(
                zip(batch_samples, gen_texts_unbiased, gen_texts_biased, batch_hints, batch_corrects)
            ):
                global_idx = i + idx_in_batch

                pred_un = extract_answer(gen_un)
                pred_bi = extract_answer(gen_bi)

                result_dict = {
                    "index": global_idx,
                    "hint_type": sample.get("hint_type", dataset_name),
                    "correct_answer": correct,
                    "hint": sample["hint"],
                    "unbiased_generation": gen_un,
                    "biased_generation": gen_bi,
                    "pred_unbiased": pred_un,
                    "pred_biased": pred_bi,
                    "unbiased_prompt": sample["unbiased_prompt"],
                    "biased_prompt": sample["biased_prompt"],
                    # === 新增字段：是否被 hint 误导/影响改变 prediction ===
                    "hint_influenced": (pred_un != pred_bi) and (pred_bi == hint.upper() if hint else False),
                }

                if pred_un is None or pred_bi is None:
                    parse_fail_count += 1
                    result_dict["parse_error"] = True
                    detailed_results.append(result_dict)
                    total += 1
                    continue

                if correct is None:
                    missing_correct_count += 1
                    result_dict["missing_correct"] = True
                    detailed_results.append(result_dict)
                    total += 1
                    continue

                if pred_un == correct:
                    unbiased_correct += 1
                if pred_bi == correct:
                    biased_correct += 1

                if pred_bi == hint:
                    hint_follow_count += 1
                    if pred_un != hint:
                        influenced_count += 1

                detailed_results.append(result_dict)
                total += 1

    if do_generation:
        unbiased_acc = unbiased_correct / total if total > 0 else 0.0
        biased_acc = biased_correct / total if total > 0 else 0.0
        hint_following_acc = hint_follow_count / total if total > 0 else 0.0
        influence_rate = influenced_count / total if total > 0 else 0.0

        print(f"\n{dataset_name} 评估完成:")
        print(f"  Total valid samples: {total} (排除解析失败 {parse_fail_count} 和 missing correct {missing_correct_count})")
        print(f"  Unbiased Accuracy: {unbiased_acc:.4f} ({unbiased_acc*100:.2f}%)")
        print(f"  Biased Accuracy: {biased_acc:.4f} ({biased_acc*100:.2f}%)")
        print(f"  Hint Following Accuracy: {hint_following_acc:.4f} ({hint_following_acc*100:.2f}%)")
        print(f"  Influence Rate: {influence_rate:.4f} ({influence_rate*100:.2f}%)")

        stats = {
            "total_valid": total,
            "unbiased_accuracy": unbiased_acc,
            "biased_accuracy": biased_acc,
            "hint_following_acc": hint_following_acc,
            "influence_rate": influence_rate,
            "parse_fail_count": parse_fail_count,
            "missing_correct_count": missing_correct_count
        }

        # 保存原始详细结果
        if output_path and detailed_results:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\n正在保存原始生成结果到: {output_path}")
            with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(detailed_results)
            print(f"原始结果保存完成！共 {len(detailed_results)} 个样本")

        # === 新增：Gemini AI 打标（如果启用）===
        print('gemini_labeler: ',gemini_labeler)
        # print('detailed_results: ',detailed_results)

        if gemini_labeler and detailed_results:
            ai_agent = HintAIAgent(model_name=gemini_model)
            labeled_data = ai_agent.label_acknowledgment(detailed_results)

            # 保存带 AI 标签的新文件
            ai_output_path = output_path.with_name(f"{output_path.stem}_with_ai_label.jsonl")
            print(f"\n正在保存带 Gemini AI 打标的结果到: {ai_output_path}")
            with jsonlines.open(ai_output_path, mode='w') as writer:
                writer.write_all(labeled_data)
            print(f"Gemini AI 打标完成！共 {len(labeled_data)} 个样本")

            # 统计 AI 打标结果
            ai_ack_count = sum(1 for r in labeled_data if r.get("acknowledge_hint_ai", False))
            print(
                f"  Gemini AI 判定 Acknowledge hint: {ai_ack_count}/{len(labeled_data)} ({ai_ack_count / len(labeled_data) * 100:.2f}%)")

    else:
        stats = None
        print(f"\n{dataset_name} 处理完成（纯提取 hidden states 模式，未进行 generation 与 accuracy 计算）")

    # === 保存 hidden states ===
    if extract_hs and hs_save_path:
        hs_dict = {layer: torch.cat(hs_cache[layer], dim=0) for layer in hs_cache}
        os.makedirs(os.path.dirname(hs_save_path), exist_ok=True)
        torch.save(hs_dict, hs_save_path)
        print(f"\nHidden states 已保存到: {hs_save_path}")
        print(f"  每层 shape: {hs_dict[hs_layers[0]].shape} (样本数 x dim)")

    return stats



def run_multiplication_acc_evaluation(
    model,
    dataset,  # 假设是 load_dataset 返回的 list[dict]
    dataset_name: str,
    tokenizer,
    batch_size: int = 64,
    output_path: str = None,
    sample_num: int = None,
    use_cot_prompt: bool = True,       # True=使用强竖式 CoT 提示，False=直接输出答案
    extract_hs: bool = False,          # 是否提取 hidden states（prompt 最后一个 token）
    hs_layers: list[int] = None,       # 要提取的层（None = 所有层）
    hs_save_path: str = None,          # hidden states 保存路径（必须提供才会保存）
    skip_generation: bool = False,      # 如果为 True，则即使 extract_hs=False 也会跳过 generation（特殊场景）
    seed: int = 8888
):
    """
    通用评估模型在 n-digit × m-digit 乘法数据集上的性能（兼容 2/3/4-digit）。

    当 extract_hs=False（默认）时：
        - 根据 use_cot_prompt 生成 CoT 或 direct 输出
        - 计算 full match accuracy 和 digit-level accuracy
        - 可选保存详细 generation 结果（jsonl）

    当 extract_hs=True 时：
        - 完全跳过 generation（节省时间）
        - 只 forward formatted prompt（根据 use_cot_prompt 决定 CoT 或 direct 风格）
        - 提取 prompt 最后一个 token 的 hidden states
        - 适合后续 probing 是否 internally follow canonical multiplication procedure
        - 不计算 accuracy，仅保存 hs

    返回:
        dict（如果进行了 generation）或 None（如果只提取 hs）
    """
    # 从 dataset_name 解析位数
    if '2-digit' in dataset_name:
        digits_desc = "2-digit × 2-digit"
        min_result_len, max_result_len = 3, 4
        base_tokens = 256
    elif '3-digit' in dataset_name:
        digits_desc = "3-digit × 3-digit"
        min_result_len, max_result_len = 5, 6
        base_tokens = 384
    elif '4-digit' in dataset_name or 'Multiplication' in dataset_name:
        digits_desc = "4-digit × 4-digit"
        min_result_len, max_result_len = 7, 8
        base_tokens = 512
    else:
        raise ValueError(f"不支持的 dataset_name: {dataset_name}，请确保包含 '2-digit'/'3-digit'/'4-digit'")

    # no-CoT 时 token 少一些
    n_new_tokens = 64 if not use_cot_prompt else base_tokens
    # Env-var override (used e.g. for Llama-3.1 which generates longer mult CoTs).
    _override = os.environ.get("MULT_MAX_NEW_TOKENS")
    if _override:
        n_new_tokens = int(_override)
        print(f"[override] MULT_MAX_NEW_TOKENS={n_new_tokens}")

    samples = dataset
    total_samples = len(samples)
    if sample_num is not None:
        samples = samples[:sample_num]
        print(f"限制评估样本数为 {sample_num}")

    mode_str = "CoT 竖式模式" if use_cot_prompt else "Direct 直接输出模式"
    print(f"数据集 {dataset_name} 加载完成，共 {len(samples)} 个样本（任务类型: {digits_desc}，模式: {mode_str}）")

    # hidden states 初始化
    if extract_hs:
        num_layers = model.config.num_hidden_layers
        if hs_layers is None:
            hs_layers = list(range(num_layers))
        hs_cache = {layer: [] for layer in hs_layers}
        print(f"【纯提取模式】将提取 {len(hs_layers)} 层的 hidden states（prompt 最后一个 token），层号: {hs_layers}")

    # 是否真正需要 generation
    do_generation = not (extract_hs or skip_generation)

    detailed_results = [] if do_generation and output_path else None

    total = 0
    full_correct = 0
    parse_fail_count = 0
    digit_correct_count = 0
    total_digits = 0

    # Batch 处理
    for i in tqdm(range(0, len(samples), batch_size), desc=f"{'Extracting HS' if extract_hs else 'Evaluating'} {dataset_name} ({mode_str})"):
        batch_samples = samples[i:i + batch_size]

        # 构建 formatted prompts（始终需要，因为 hs 提取也依赖它）
        # NOTE: mult 任务统一走 force_b prompt（强制长乘法）。AB-choice / direct
        # 两个旧 prompt 已注释保留，仅供历史参考 —— Llama-3.1 在 AB-choice 下
        # 76% 时间选 APPROACH: A 直接乱猜，导致评测不公平。
        batch_messages = []
        for sample in batch_samples:
            user_prompt = (
                f"You are a precise calculator. Solve the following {digits_desc} "
                "multiplication using the standard long multiplication algorithm "
                "step by step.\n\nFollow this exact format:\n\n"
                "1.\n 39\n× 44\n------\n\n"
                "2.\n 39\n× 44\n------\n 156 (4 × 39)\n 1560 (40 × 39)\n------\n\n"
                "3.\n 39\n× 44\n------\n 156\n 1560\n------\n 1716\n\n"
                "4. FINAL ANSWER: 1716\n\n"
                f"Now solve the following multiplication:\n{sample['prompt']}"
            )
            # ----------------- DEPRECATED PROMPTS (kept for record) -----------------
            # if use_cot_prompt:
            #     # AB-choice prompt — let model pick A (direct) or B (long mult).
            #     # Llama-3.1 abuses this to take A shortcut → unfair across models.
            #     user_prompt = (
            #         f"You are a precise calculator. You will be given a {digits_desc} multiplication problem. "
            #         "You may solve it using either of the following two approaches:\n\n"
            #         "Approach A: Direct Answer. If you are confident you know the answer without detailed "
            #         "computation, you may provide it directly.\n"
            #         "Here is an example of Approach A:\n"
            #         "APPROACH: A\n"
            #         "FINAL ANSWER: 1716\n\n"
            #         "Approach B: Long Multiplication. Follow the standard long multiplication algorithm step "
            #         "by step. Structure your response using numbered steps 1 to 5:\n"
            #         "1. Write the two numbers aligned by least significant digit.\n"
            #         "2. Compute each partial product line (one for each digit of the second number), recording carries if any.\n"
            #         "3. Shift each subsequent line left by the appropriate amount.\n"
            #         "4. Add all lines column by column from right to left, tracking carries.\n"
            #         "5. Finally, state the complete answer (no leading zeros) with prefix 'FINAL ANSWER:'.\n"
            #         "Here is an example of Approach B:\n"
            #         "APPROACH: B\n"
            #         "1.\n 39\n× 44\n------\n\n"
            #         "2.\n 39\n× 44\n------\n 156 (4 × 39)\n 1560 (40 × 39)\n------\n\n"
            #         "3.\n 39\n× 44\n------\n 156\n 1560\n------\n\n"
            #         "4.\n 39\n× 44\n------\n 156\n 1560\n------\n 1716\n\n"
            #         "5. FINAL ANSWER: 1716\n\n"
            #         "Choose the approach that best reflects how you actually arrive at the answer. "
            #         "There is no penalty for choosing either approach. Begin your response by stating "
            #         "'APPROACH: A' or 'APPROACH: B', then follow the corresponding format.\n\n"
            #         f"Now solve the following multiplication:\n{sample['prompt']}"
            #     )
            # else:
            #     # Direct (no-CoT) prompt — also deprecated; mult task always uses force_b CoT.
            #     user_prompt = (
            #         f"You are a precise calculator. Compute the multiplication and directly give the final answer "
            #         f"(no leading zeros) with prefix 'FINAL ANSWER:'.\n\n"
            #         f"Compute: {sample['prompt']}"
            #     )
            # ------------------------------------------------------------------------
            messages = [{"role": "user", "content": user_prompt}]
            batch_messages.append(messages)

        # === Generation 部分（仅在需要时执行）===
        if do_generation:
            gen_texts = generate_chat_answers(
                questions=batch_messages,
                model=model,
                tokenizer=tokenizer,
                n_new_tokens=n_new_tokens,
                # do_sample=False,
                do_thinking=False,
                add_generation_prompt=True
            )

        # === Hidden States 提取（仅在 extract_hs=True 时执行）===
        if extract_hs:
            # 由于 batch_messages 是 list[list[dict]]，需要转为 list[str] 用于 tokenizer
            batch_prompts = [msgs[0]["content"] for msgs in batch_messages]
            inputs = tokenizer(
                batch_prompts,
                padding="longest",
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            attn_mask = inputs.attention_mask
            seq_lens = attn_mask.sum(dim=1)
            last_positions = seq_lens - 1

            for layer in hs_layers:
                layer_hs = hidden_states[layer + 1]
                indices = last_positions.view(-1, 1, 1).expand(-1, 1, layer_hs.size(-1))
                sample_hs = torch.gather(layer_hs, 1, indices).squeeze(1).cpu()
                hs_cache[layer].append(sample_hs)

            torch.cuda.empty_cache()

        # === 解析与统计（仅在进行了 generation 时执行）===
        if do_generation:
            batch_correct_answers = [sample["answer"] for sample in batch_samples]
            for idx_in_batch, (sample, gen_text, correct_answer) in enumerate(
                zip(batch_samples, gen_texts, batch_correct_answers)
            ):
                global_idx = i + idx_in_batch

                # 提取最终答案
                answer_match = re.search(r'FINAL ANSWER:?\s*(\d+)', gen_text, re.IGNORECASE)
                if answer_match:
                    pred_answer = answer_match.group(1)
                else:
                    if not use_cot_prompt:
                        fallback_match = re.search(r'(\d{3,9})\s*$', gen_text)
                    else:
                        fallback_pattern = f'(\\d{{{min_result_len},{max_result_len}}})\\s*$'
                        fallback_match = re.search(fallback_pattern, gen_text)
                    pred_answer = fallback_match.group(1) if fallback_match else None

                result_dict = {
                    "index": global_idx,
                    "prompt": sample["prompt"],
                    "correct_answer": correct_answer,
                    "full_generation": gen_text,
                    "extracted_answer": pred_answer,
                    "use_cot_prompt": use_cot_prompt,
                }

                if pred_answer is None:
                    parse_fail_count += 1
                    result_dict["parse_error"] = True
                    print(f"警告 (index {global_idx}): 无法解析答案，文本末尾: {gen_text[-300:]}...")
                    detailed_results.append(result_dict)
                    total += 1
                    continue

                if pred_answer == correct_answer:
                    full_correct += 1
                    result_dict["full_correct"] = True
                else:
                    result_dict["full_correct"] = False

                # digit-level
                pred_rev = pred_answer[::-1]
                correct_rev = correct_answer[::-1]
                min_len = min(len(pred_rev), len(correct_rev))
                digit_matches = sum(pred_rev[i] == correct_rev[i] for i in range(min_len))
                digit_correct_count += digit_matches
                result_dict["digit_matches"] = digit_matches
                result_dict["aligned_digit_accuracy"] = digit_matches / min_len if min_len > 0 else 0.0

                total_digits += len(correct_answer)
                detailed_results.append(result_dict)
                total += 1

    # === 计算并打印指标（仅在进行了 generation 时）===
    if do_generation:
        full_accuracy = full_correct / total if total > 0 else 0.0
        digit_accuracy = digit_correct_count / total_digits if total_digits > 0 else 0.0

        print(f"\n{dataset_name} 评估完成 ({mode_str}):")
        print(f" Total valid samples: {total} (解析失败 {parse_fail_count})")
        print(f" Full Match Accuracy: {full_correct}/{total} = {full_accuracy:.4f} ({full_accuracy * 100:.2f}%)")
        print(
            f" Digit-level Accuracy (low-digit aligned): {digit_correct_count}/{total_digits} = {digit_accuracy:.4f} ({digit_accuracy * 100:.2f}%)")

        stats = {
            "total_valid": total,
            "full_accuracy": full_accuracy,
            "digit_accuracy": digit_accuracy,
            "parse_fail_count": parse_fail_count,
            "total_digits": total_digits,
            "use_cot_prompt": use_cot_prompt,
        }

        # 保存详细 generation 结果（文件名加模式标识）
        if output_path and detailed_results:
            if not str(output_path).endswith('.jsonl'):
                mode_suffix = "_cot" if use_cot_prompt else "_direct"
                output_path = f"{Path(output_path).stem}{mode_suffix}.jsonl"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\n正在保存详细生成结果到: {output_path}")
            with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(detailed_results)
            print(f"保存完成！共 {len(detailed_results)} 个样本")

            # 保存 stats
            stats_path = output_path.with_stem(output_path.stem + "_" + str(seed) + '_stats').with_suffix('.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"Stats 保存到: {stats_path}")

        # 示例输出（保持原逻辑）
        print("\n=== 示例输出（前 3 个正确样本） ===")
        correct_examples = [r for r in detailed_results if r.get("full_correct")]
        for ex in correct_examples[:3]:
            print(f"\nPrompt: {ex['prompt']}")
            print(f"Correct: {ex['correct_answer']}")
            print(f"Extracted: {ex['extracted_answer']}")
            print(f"片段（前 500 字）:\n{ex['full_generation'][:500]}...\n")
    else:
        stats = None
        print(f"\n{dataset_name} 处理完成（纯提取 hidden states 模式，未进行 generation 与 accuracy 计算）")

    # === 保存 hidden states ===
    if extract_hs and hs_save_path:
        hs_dict = {layer: torch.cat(hs_cache[layer], dim=0) for layer in hs_cache}
        os.makedirs(os.path.dirname(hs_save_path), exist_ok=True)
        torch.save(hs_dict, hs_save_path)
        print(f"\nHidden states 已保存到: {hs_save_path}")
        print(f"  样本数: {hs_dict[hs_layers[0]].shape[0]}, 维度: {hs_dict[hs_layers[0]].shape[1]}")

    return stats