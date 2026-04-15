"""
Evaluation utilities for LLM-based causal DAG selection.

Metrics
-------
- **Top-1 accuracy**      : fraction of examples where the LLM selected the
                            exact ground-truth graph.
- **Candidate rank**      : 0-based rank of the selected graph within RASL
                            candidates (lower = better; 0 = correct).
- **Edge-F1**             : token-level F1 on the predicted vs. ground-truth
                            directed and bidirected edge sets.
- **Structural Hamming**  : number of edge-type mismatches (directed/bidirected)
                            between prediction and ground truth.

Usage
-----
    from inference import LLMCausalSelector, SelectionEvaluator
    from data_generation import CausalDatasetGenerator

    gen   = CausalDatasetGenerator(n_nodes=5, seed=0)
    evals = gen.generate(100)

    sel = LLMCausalSelector.from_pretrained("outputs/lora-causal-select")
    ev  = SelectionEvaluator(selector=sel)
    results = ev.evaluate(evals)
    ev.print_report(results)
"""

import logging
from typing import Optional

import numpy as np

from .selector import LLMCausalSelector

logger = logging.getLogger(__name__)


class SelectionEvaluator:
    """Evaluate a trained LLMCausalSelector on a set of labelled examples.

    Parameters
    ----------
    selector : LLMCausalSelector
        Loaded, fine-tuned selector.
    """

    def __init__(self, selector: LLMCausalSelector):
        self.selector = selector

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self, examples: list[dict]) -> dict:
        """Evaluate the selector on a list of labelled examples.

        Parameters
        ----------
        examples : list of dicts from CausalDatasetGenerator.generate()
                   Each must have 'metadata' with keys:
                       gt_graph, g_u, u, n_candidates, gt_index, candidates (optional)

        Returns
        -------
        dict with keys: accuracy, mean_edge_f1, mean_shd, per_example
        """
        per_example = []
        for ex in examples:
            meta = ex.get("metadata", {})
            g_u = meta.get("g_u")
            gt_graph = meta.get("gt_graph")
            gt_index = meta.get("gt_index")
            u = meta.get("u", 2)
            candidates = meta.get("candidates")

            if g_u is None or gt_graph is None or gt_index is None:
                logger.warning("Skipping example missing metadata fields")
                continue

            # If candidates not stored in metadata, re-run RASL (slower)
            if candidates is None:
                logger.debug("candidates not in metadata; skipping")
                continue

            pred_graph, pred_index = self.selector.select(g_u, candidates, u)

            correct = pred_index == gt_index
            edge_f1 = self._edge_f1(pred_graph, gt_graph) if pred_graph is not None else 0.0
            shd = self._shd(pred_graph, gt_graph) if pred_graph is not None else float("inf")

            per_example.append({
                "correct": correct,
                "pred_index": pred_index,
                "gt_index": gt_index,
                "edge_f1": edge_f1,
                "shd": shd,
                "n_candidates": meta.get("n_candidates"),
            })

        if not per_example:
            return {"accuracy": None, "mean_edge_f1": None, "mean_shd": None, "per_example": []}

        accuracy = np.mean([e["correct"] for e in per_example])
        mean_f1 = np.mean([e["edge_f1"] for e in per_example])
        mean_shd = np.mean([e["shd"] for e in per_example if np.isfinite(e["shd"])])

        return {
            "accuracy": float(accuracy),
            "mean_edge_f1": float(mean_f1),
            "mean_shd": float(mean_shd),
            "n_examples": len(per_example),
            "per_example": per_example,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(results: dict) -> None:
        n = results.get("n_examples", 0)
        print(f"Evaluated on {n} examples")
        print(f"  Top-1 Accuracy  : {results['accuracy']:.3f}")
        print(f"  Mean Edge-F1    : {results['mean_edge_f1']:.3f}")
        print(f"  Mean SHD        : {results['mean_shd']:.2f}")

        # Accuracy vs. random baseline
        avg_cands = np.mean([e["n_candidates"] for e in results["per_example"] if e["n_candidates"]])
        if avg_cands > 0:
            print(f"  Random baseline : {1.0 / avg_cands:.3f}  (avg {avg_cands:.1f} candidates)")

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_f1(pred: Optional[dict], gt: dict) -> float:
        """F1 over the full set of (src, tgt, edge_type) triples."""
        if pred is None:
            return 0.0
        pred_edges = _edge_set(pred)
        gt_edges = _edge_set(gt)
        if not gt_edges:
            return 1.0 if not pred_edges else 0.0
        tp = len(pred_edges & gt_edges)
        fp = len(pred_edges - gt_edges)
        fn = len(gt_edges - pred_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _shd(pred: Optional[dict], gt: dict) -> float:
        """Structural Hamming Distance: number of edge-type mismatches."""
        if pred is None:
            return float(len(_edge_set(gt)))
        pred_edges = _edge_set(pred)
        gt_edges = _edge_set(gt)
        return float(len(pred_edges.symmetric_difference(gt_edges)))


def _edge_set(graph: dict) -> frozenset:
    edges = set()
    for src, targets in graph.items():
        for tgt, etype in targets.items():
            edges.add((src, tgt, etype))
    return frozenset(edges)
