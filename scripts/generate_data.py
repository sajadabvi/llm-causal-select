"""
Generate synthetic training and evaluation data.

Usage
-----
    python scripts/generate_data.py --n_train 2000 --n_eval 200 --n_nodes 5

Output
------
    data/train.jsonl
    data/eval.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation import CausalDatasetGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_eval",  type=int, default=200)
    p.add_argument("--n_nodes", type=int, default=5)
    p.add_argument("--max_urate", type=int, default=3)
    p.add_argument("--capsize",   type=int, default=20)
    p.add_argument("--degree",    type=int, default=3)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default="data")
    return p.parse_args()


def main():
    args = parse_args()

    gen = CausalDatasetGenerator(
        n_nodes=args.n_nodes,
        max_urate=args.max_urate,
        capsize=args.capsize,
        degree=args.degree,
        seed=args.seed,
    )

    print(f"Generating {args.n_train} training examples...")
    train_ex = gen.generate(args.n_train)
    gen.save(train_ex, f"{args.out_dir}/train.jsonl")

    print(f"Generating {args.n_eval} eval examples...")
    eval_ex = gen.generate(args.n_eval)
    gen.save(eval_ex, f"{args.out_dir}/eval.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
