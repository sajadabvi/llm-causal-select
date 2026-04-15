"""
Run evaluation on a saved LoRA adapter using labelled examples.

Usage
-----
    python scripts/run_eval.py \\
        --adapter_path outputs/lora-causal-select \\
        --eval_file    data/eval.jsonl \\
        --n_examples   100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import LLMCausalSelector, SelectionEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--eval_file",    required=True)
    p.add_argument("--n_examples",   type=int, default=None)
    p.add_argument("--temperature",  type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading selector from {args.adapter_path} ...")
    selector = LLMCausalSelector.from_pretrained(
        args.adapter_path,
        temperature=args.temperature,
    )

    print(f"Loading eval examples from {args.eval_file} ...")
    examples = load_jsonl(args.eval_file)
    if args.n_examples:
        examples = examples[: args.n_examples]
    print(f"Evaluating on {len(examples)} examples")

    evaluator = SelectionEvaluator(selector)
    results = evaluator.evaluate(examples)
    SelectionEvaluator.print_report(results)


if __name__ == "__main__":
    main()
