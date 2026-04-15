"""
LLM-based causal DAG selector — inference interface.

Drop-in complement to the gunfolds RASL pipeline:

    from gunfolds.solvers.clingo_rasl import drasl
    from gunfolds.utils import bfutils
    from inference import LLMCausalSelector

    g_u   = bfutils.undersample(gt_graph, u=2)
    cands = drasl([g_u], urate=6)

    selector = LLMCausalSelector.from_pretrained("outputs/lora-causal-select")
    selected_graph, selected_index = selector.select(g_u, cands, u=2)

The selector generates the LLM prompt via GraphFormatter, runs inference,
parses the completion, and returns both the selected gunfolds graph dict
and the 0-based candidate index.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_generation.formatter import GraphFormatter

logger = logging.getLogger(__name__)

_INSTRUCT_TEMPLATE = "<s>[INST] {prompt} [/INST]"
_MAX_NEW_TOKENS = 128
_DEFAULT_TEMPERATURE = 0.1


class LLMCausalSelector:
    """Load a fine-tuned LoRA adapter and use it to select DAGs from RASL output.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The loaded (base + LoRA merged or PEFT-wrapped) model.
    tokenizer : AutoTokenizer
        Matching tokenizer.
    device : str
        Torch device string ('cuda', 'cpu', 'mps').
    temperature : float
        Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
        temperature: float = _DEFAULT_TEMPERATURE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.formatter = GraphFormatter()

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        base_model_name: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        merge_weights: bool = False,
    ) -> "LLMCausalSelector":
        """Load a saved LoRA adapter (and optionally the base model).

        Parameters
        ----------
        adapter_path : str
            Directory produced by ``finetune.train`` containing the LoRA
            adapter_config.json and adapter_model.bin (or safetensors).
        base_model_name : Optional[str]
            Base model name/path.  If None, read from adapter_config.json.
        device : Optional[str]
            Torch device.  Auto-detected if None.
        merge_weights : bool
            If True, merge LoRA weights into the base model and unload the
            adapter (faster inference, no PEFT dependency at runtime).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        if base_model_name is None:
            import json, os
            cfg_path = os.path.join(adapter_path, "adapter_config.json")
            with open(cfg_path) as f:
                base_model_name = json.load(f)["base_model_name_or_path"]

        logger.info("Loading base model: %s", base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )

        logger.info("Loading LoRA adapter from: %s", adapter_path)
        model = PeftModel.from_pretrained(base_model, adapter_path)

        if merge_weights:
            logger.info("Merging LoRA weights into base model")
            model = model.merge_and_unload()

        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=device, temperature=temperature)

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select(
        self,
        g_u: dict,
        candidates: list[dict],
        u: int,
    ) -> tuple[Optional[dict], Optional[int]]:
        """Select the most likely causal-timescale DAG from RASL candidates.

        Parameters
        ----------
        g_u        : observed undersampled graph (gunfolds dict)
        candidates : list of candidate graphs from drasl()
        u          : undersampling rate

        Returns
        -------
        (selected_graph, selected_index)
            selected_graph  – the chosen gunfolds graph dict (or None on parse failure)
            selected_index  – 0-based index into candidates (or None on parse failure)
        """
        prompt = self.formatter.to_prompt(g_u, candidates, u, include_system=True)
        prompt_text = _INSTRUCT_TEMPLATE.format(prompt=prompt)

        completion = self._generate(prompt_text)
        logger.debug("LLM completion:\n%s", completion)

        idx = self.formatter.completion_to_index(completion)
        if idx is None or idx < 0 or idx >= len(candidates):
            logger.warning("Could not parse valid index from completion: %r", completion)
            return None, None

        return candidates[idx], idx

    def select_batch(
        self,
        examples: list[tuple[dict, list[dict], int]],
    ) -> list[tuple[Optional[dict], Optional[int]]]:
        """Run select() on a batch of (g_u, candidates, u) tuples."""
        return [self.select(g_u, cands, u) for g_u, cands, u in examples]

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(self, prompt_text: str) -> str:
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
