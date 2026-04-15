"""
Hyperparameter dataclasses for LoRA fine-tuning.

These mirror the fields accepted by HuggingFace PEFT and Transformers
Trainer, and can be loaded from configs/lora_config.yaml.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class LoRAConfig:
    """LoRA adapter configuration (maps to peft.LoraConfig)."""

    r: int = 16
    """Adapter rank — controls capacity. 8–64 typical. Higher = more params."""

    lora_alpha: int = 32
    """LoRA scaling factor. Good default: alpha = 2 * r."""

    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    """Which attention projection layers to adapt."""

    lora_dropout: float = 0.05
    """Dropout probability applied to LoRA adapter weights."""

    bias: str = "none"
    """Whether to train bias parameters ('none', 'all', 'lora_only')."""

    task_type: str = "CAUSAL_LM"
    """PEFT task type. Always CAUSAL_LM for decoder-only fine-tuning."""


@dataclass
class TrainingConfig:
    """Training loop configuration (maps to transformers.TrainingArguments)."""

    # Model
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"
    """Base model to fine-tune. Llama-3-8B or Mistral-7B recommended."""

    use_4bit: bool = True
    """Enable QLoRA: load base model in 4-bit NF4 quantization."""

    bnb_4bit_compute_dtype: str = "bfloat16"
    """Compute dtype for 4-bit layers ('bfloat16' or 'float16')."""

    # Data
    train_file: str = "data/train.jsonl"
    eval_file: str = "data/eval.jsonl"
    max_seq_length: int = 1024
    """Truncation length in tokens. Prompts + completions must fit here."""

    # Output
    output_dir: str = "outputs/lora-causal-select"
    logging_dir: str = "outputs/logs"

    # Optimisation
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    """Effective batch size = per_device * accumulation * n_gpus."""

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Regularisation
    weight_decay: float = 0.01

    # Logging / eval
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    logging_steps: int = 10
    report_to: str = "none"
    """Set to 'wandb' or 'mlflow' to enable experiment tracking."""

    # Misc
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 0
    seed: int = 42

    # Optional: push to HF Hub after training
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


def load_from_yaml(path: str) -> tuple[LoRAConfig, TrainingConfig]:
    """Load both configs from a single YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    lora_raw = raw.get("lora", {})
    train_raw = raw.get("training", {})

    lora = LoRAConfig(**lora_raw)
    train = TrainingConfig(**train_raw)
    return lora, train
