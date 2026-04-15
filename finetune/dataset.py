"""
PyTorch Dataset for causal DAG selection fine-tuning.

Each record in the JSONL file has two fields:
    ``prompt``     – the full input text (system + observed graph + candidates)
    ``completion`` – the expected output text (selected index + edge list)

At training time the two are concatenated and the prompt tokens are masked
(label = -100) so the model only computes loss on the completion tokens.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CausalGraphDataset(Dataset):
    """Token-level dataset for causal graph DAG-selection fine-tuning.

    Parameters
    ----------
    path : str
        Path to a JSONL file produced by ``CausalDatasetGenerator.save()``.
    tokenizer : PreTrainedTokenizer
        Tokenizer matching the base model.
    max_length : int
        Maximum sequence length (tokens).  Sequences are truncated.
    prompt_template : Optional[str]
        If provided, wrap the raw prompt string with this template.
        Use ``{prompt}`` as the placeholder.  Useful for chat models that
        expect a specific instruction format.
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        prompt_template: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.records = self._load(path)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        prompt = self._apply_template(record["prompt"])
        completion = record["completion"]

        # Tokenize prompt and full sequence separately so we can mask prompt
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        full_text = prompt + "\n" + completion
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        # Append EOS token if it fits
        if len(full_ids) < self.max_length:
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                full_ids = full_ids + [eos]

        labels = list(full_ids)
        # Mask prompt portion with -100 (ignored in cross-entropy loss)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> list[dict]:
        records = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _apply_template(self, prompt: str) -> str:
        if self.prompt_template is None:
            return prompt
        return self.prompt_template.format(prompt=prompt)


def collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        labels[i, :seq_len] = b["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
