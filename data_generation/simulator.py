"""
Synthetic training data generator for LLM-based causal DAG selection.

Uses the gunfolds causal simulator to produce ground-truth DAGs, simulate
undersampling, run RASL to enumerate equivalence-class candidates, then
package everything as (prompt, completion) pairs for LoRA fine-tuning.

Pipeline per example
--------------------
1. randomDAG(N)              → ground-truth causal-timescale graph G*
2. undersample(G*, u)        → observed undersampled graph G_u
3. drasl([G_u], urate=u_max) → set of candidate graphs C
4. format(G_u, C, G*)        → (prompt, completion) text pair
"""

import json
import random
import logging
from pathlib import Path
from typing import Optional

from gunfolds.utils import graphkit, bfutils
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds import conversions

from .formatter import GraphFormatter

logger = logging.getLogger(__name__)


class CausalDatasetGenerator:
    """Generate synthetic (prompt, completion) pairs for LLM fine-tuning.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in each generated DAG.
    max_urate : int
        Maximum undersampling rate to consider when calling RASL.
    capsize : int
        Maximum number of RASL candidate solutions to retain per example.
    rasl_timeout : int
        Per-example timeout (seconds) passed to drasl. 0 = unlimited.
    degree : int
        Average node degree for randomDAG generation.
    seed : Optional[int]
        Global random seed for reproducibility.
    """

    def __init__(
        self,
        n_nodes: int = 5,
        max_urate: int = 3,
        capsize: int = 20,
        rasl_timeout: int = 60,
        degree: int = 3,
        seed: Optional[int] = None,
    ):
        self.n_nodes = n_nodes
        self.max_urate = max_urate
        self.capsize = capsize
        self.rasl_timeout = rasl_timeout
        self.degree = degree
        self.formatter = GraphFormatter()

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_examples: int) -> list[dict]:
        """Generate *n_examples* training examples.

        Returns a list of dicts, each with keys:
            ``prompt``       – LLM input text
            ``completion``   – LLM target text (ground-truth DAG edges)
            ``metadata``     – dict with raw graph objects for debugging
        """
        examples = []
        attempts = 0
        max_attempts = n_examples * 5

        while len(examples) < n_examples and attempts < max_attempts:
            attempts += 1
            example = self._generate_one()
            if example is not None:
                examples.append(example)
                logger.info(
                    "Generated example %d/%d (attempt %d)",
                    len(examples),
                    n_examples,
                    attempts,
                )

        if len(examples) < n_examples:
            logger.warning(
                "Only generated %d/%d examples after %d attempts",
                len(examples),
                n_examples,
                attempts,
            )
        return examples

    def save(self, examples: list[dict], path: str) -> None:
        """Save examples to a JSONL file (one JSON object per line)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for ex in examples:
                # Strip raw graph objects — not JSON-serialisable
                record = {"prompt": ex["prompt"], "completion": ex["completion"]}
                f.write(json.dumps(record) + "\n")
        logger.info("Saved %d examples to %s", len(examples), path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(self) -> Optional[dict]:
        """Attempt to produce one training example. Returns None on failure."""
        # 1. Sample ground-truth DAG and convert to gunfolds format
        try:
            nx_dag = graphkit.randomDAG(self.n_nodes, degree=self.degree, connected=True)
            gt_graph = conversions.nx2graph(nx_dag)
        except Exception as e:
            logger.debug("randomDAG failed: %s", e)
            return None

        # 2. Pick a random undersampling rate
        u = random.randint(2, self.max_urate)

        # 3. Undersample
        try:
            g_u = bfutils.undersample(gt_graph, u)
        except Exception as e:
            logger.debug("undersample failed: %s", e)
            return None

        # Reject empty observations (all edges wiped out)
        if not bfutils.g2num(g_u):
            return None

        # 4. Run RASL to get the equivalence class
        try:
            candidates = drasl(
                [g_u],
                capsize=self.capsize,
                urate=min(self.max_urate, 3 * self.n_nodes + 1),
                timeout=self.rasl_timeout,
                scc=False,
                weighted=False,
                configuration="crafty",
            )
        except Exception as e:
            logger.debug("drasl failed: %s", e)
            return None

        # Need at least 2 candidates to make selection non-trivial
        if len(candidates) < 2:
            return None

        # 5. Find the index of the ground-truth graph in the candidate list
        gt_index = self._find_gt_index(gt_graph, candidates)
        if gt_index is None:
            # Ground truth not in RASL output — skip (can happen at capsize limit)
            logger.debug("GT not found in candidates; skipping example")
            return None

        # 6. Format as prompt/completion
        prompt = self.formatter.to_prompt(g_u, candidates, u)
        completion = self.formatter.to_completion(gt_graph, gt_index)

        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "gt_graph": gt_graph,
                "g_u": g_u,
                "u": u,
                "n_candidates": len(candidates),
                "gt_index": gt_index,
            },
        }

    @staticmethod
    def _find_gt_index(gt_graph: dict, candidates: list[dict]) -> Optional[int]:
        """Return the 0-based index of gt_graph in candidates, or None."""
        gt_edges = CausalDatasetGenerator._edge_set(gt_graph)
        for i, cand in enumerate(candidates):
            if CausalDatasetGenerator._edge_set(cand) == gt_edges:
                return i
        return None

    @staticmethod
    def _edge_set(graph: dict) -> frozenset:
        """Canonical edge-set representation for equality checks."""
        edges = set()
        for src, targets in graph.items():
            for tgt, etype in targets.items():
                edges.add((src, tgt, etype))
        return frozenset(edges)
