# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""DPO training script for CIA (paper §5).

Trains on a preference-pair dataset produced by
`scripts/cia/build_dpo_dataset.py`. Dataset columns:
    prompt, chosen, rejected
which is the standard TRL `DPOTrainer` schema.

Mirrors `grpo.py` for environment setup (torch.load patch, logging,
wandb), but uses TRL's `DPOTrainer` and `DPOConfig`.

Usage:
    accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml \\
        src/open_r1/dpo.py \\
        --config recipes/CIA/dpo/hint.yaml
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

import gc
import torch

# Same torch.load weights_only=False patch as grpo.py for DeepSpeed ckpt resume.
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum])
except Exception:
    pass

_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe

from open_r1.configs import DPOConfig, ScriptArguments
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from transformers import AutoModelForCausalLM
from trl import DPOTrainer, ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


def main(script_args: ScriptArguments,
         training_args: DPOConfig,
         model_args: ModelConfig):
    set_seed(training_args.seed)

    handlers = [logging.StreamHandler(sys.stdout)]
    if training_args.local_rank in [-1, 0]:
        handlers.append(logging.FileHandler("dpo_training.log", mode="w",
                                            encoding="utf-8"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu} distributed: {bool(training_args.local_rank != -1)}, "
        f"16-bit: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # ── dataset ───────────────────────────────────────────────────────
    dataset = get_dataset(script_args)

    # Optional dev sub-sampling
    _dev_frac = os.environ.get("CIA_DEV_FRAC")
    if _dev_frac:
        denom = int(_dev_frac)
        print(f"WARN: CIA_DEV_FRAC={denom} — using 1/{denom} of every split")
        for split in dataset.keys():
            n = len(dataset[split])
            dataset[split] = dataset[split].select(range(max(1, n // denom)))

    # Sanity check: DPO needs prompt / chosen / rejected columns.
    needed = {"prompt", "chosen", "rejected"}
    train_cols = set(dataset[script_args.dataset_train_split].column_names)
    if not needed.issubset(train_cols):
        raise ValueError(
            f"DPO dataset missing required columns. Need {needed}, "
            f"have {train_cols}. Did you build it with build_dpo_dataset.py?"
        )

    # ── tokenizer / model ─────────────────────────────────────────────
    tokenizer = get_tokenizer(model_args, training_args)

    # Qwen3 thinking-mode disable (same patch as grpo.py).
    if "<think>" in (tokenizer.chat_template or ""):
        _orig_apply = tokenizer.apply_chat_template
        def _no_think(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_apply(*args, **kwargs)
        tokenizer.apply_chat_template = _no_think
        logger.info("Patched apply_chat_template to disable Qwen3 thinking")

    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Explicitly load reference model — TRL's create_reference_model()
    # (used when ref_model=None) is incompatible with DeepSpeed ZeRO-3.
    # We load from the same base model checkpoint as the policy.
    logger.info("*** Loading reference model (separate instance for DPO) ***")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation or "sdpa",
        trust_remote_code=model_args.trust_remote_code,
    )
    ref_model.eval()

    # ── trainer ───────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split]
                      if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # ── train ─────────────────────────────────────────────────────────
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ── save ─────────────────────────────────────────────────────────
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["cia", "dpo", "open-r1"]})
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
