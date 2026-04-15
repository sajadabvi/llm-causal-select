"""
LoRA fine-tuning entry-point for causal DAG selection.

Usage
-----
    python -m finetune.train --config configs/lora_config.yaml

    # Override individual fields
    python -m finetune.train \\
        --config configs/lora_config.yaml \\
        --train_file data/train.jsonl \\
        --eval_file data/eval.jsonl \\
        --output_dir outputs/run1

What happens
------------
1. Load base model in 4-bit (QLoRA) or full precision.
2. Wrap with PEFT LoRA adapters (q_proj + v_proj by default).
3. Fine-tune with HuggingFace Trainer.
4. Save the LoRA adapter weights to output_dir.

The saved adapter is small (~50–200 MB for rank-16 Mistral-7B).
Load it back with LLMCausalSelector for inference.
"""

import argparse
import logging
import os
from functools import partial

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from .config import LoRAConfig, TrainingConfig, load_from_yaml
from .dataset import CausalGraphDataset, collate_fn

logger = logging.getLogger(__name__)

# Mistral / Llama instruction-format wrapper
_INSTRUCT_TEMPLATE = "<s>[INST] {prompt} [/INST]"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune for causal DAG selection")
    p.add_argument("--config", required=True, help="Path to lora_config.yaml")
    p.add_argument("--train_file", default=None)
    p.add_argument("--eval_file", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--report_to", default=None, choices=["none", "wandb", "mlflow"])
    return p.parse_args()


def build_model_and_tokenizer(
    lora_cfg: LoRAConfig,
    train_cfg: TrainingConfig,
) -> tuple:
    """Load the base model and tokenizer, apply LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: 4-bit quantization via bitsandbytes
    if train_cfg.use_4bit:
        compute_dtype = (
            torch.bfloat16
            if train_cfg.bnb_4bit_compute_dtype == "bfloat16"
            else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            train_cfg.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            train_cfg.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_datasets(
    train_cfg: TrainingConfig,
    tokenizer,
) -> tuple[CausalGraphDataset, CausalGraphDataset]:
    """Build train and eval datasets."""
    train_ds = CausalGraphDataset(
        path=train_cfg.train_file,
        tokenizer=tokenizer,
        max_length=train_cfg.max_seq_length,
        prompt_template=_INSTRUCT_TEMPLATE,
    )
    eval_ds = CausalGraphDataset(
        path=train_cfg.eval_file,
        tokenizer=tokenizer,
        max_length=train_cfg.max_seq_length,
        prompt_template=_INSTRUCT_TEMPLATE,
    )
    return train_ds, eval_ds


def build_training_args(train_cfg: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=train_cfg.output_dir,
        logging_dir=train_cfg.logging_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        eval_strategy=train_cfg.eval_strategy,
        save_strategy=train_cfg.save_strategy,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        seed=train_cfg.seed,
        push_to_hub=train_cfg.push_to_hub,
        hub_model_id=train_cfg.hub_model_id,
        remove_unused_columns=False,
    )


def train(lora_cfg: LoRAConfig, train_cfg: TrainingConfig) -> None:
    set_seed(train_cfg.seed)
    os.makedirs(train_cfg.output_dir, exist_ok=True)

    logger.info("Loading model: %s", train_cfg.model_name_or_path)
    model, tokenizer = build_model_and_tokenizer(lora_cfg, train_cfg)

    logger.info("Building datasets")
    train_ds, eval_ds = build_datasets(train_cfg, tokenizer)
    logger.info("Train examples: %d  |  Eval examples: %d", len(train_ds), len(eval_ds))

    pad_id = tokenizer.pad_token_id
    _collate = partial(collate_fn, pad_token_id=pad_id)

    trainer = Trainer(
        model=model,
        args=build_training_args(train_cfg),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", train_cfg.output_dir)
    model.save_pretrained(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)
    logger.info("Done.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    lora_cfg, train_cfg = load_from_yaml(args.config)

    # Allow CLI overrides
    if args.train_file:
        train_cfg.train_file = args.train_file
    if args.eval_file:
        train_cfg.eval_file = args.eval_file
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.num_train_epochs is not None:
        train_cfg.num_train_epochs = args.num_train_epochs
    if args.report_to:
        train_cfg.report_to = args.report_to

    train(lora_cfg, train_cfg)


if __name__ == "__main__":
    main()
