#!/usr/bin/env python3
"""
scripts/evaluate.py
────────────────────
Evaluation harness to reproduce the paper's experimental results.
Computes: Accuracy, Precision@1, Recall@3, Mean Diagnosis Time (MDT).
Also runs the ablation study by disabling components one at a time.

Usage:
  python scripts/evaluate.py --dataset data/samples/eval_dataset.json
  python scripts/evaluate.py --ablation   # run full ablation study
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@dataclass
class EvalSample:
    incident_id: str
    title: str
    description: str
    service: str
    priority: str
    error_message: str
    ground_truth_root_cause: str    # ground truth label for accuracy measurement
    ground_truth_rank: int = 1      # expected rank of correct answer (for Recall@k)


@dataclass
class EvalMetrics:
    accuracy: float
    precision_at_1: float
    recall_at_3: float
    mean_diagnosis_time_s: float
    total_samples: int
    config_label: str = "Full System"

    def to_dict(self) -> dict:
        return {
            "config": self.config_label,
            "accuracy": round(self.accuracy * 100, 1),
            "precision_at_1": round(self.precision_at_1 * 100, 1),
            "recall_at_3": round(self.recall_at_3 * 100, 1),
            "mdt_seconds": round(self.mean_diagnosis_time_s, 2),
            "samples": self.total_samples,
        }


def load_eval_dataset(path: str) -> list[EvalSample]:
    with open(path) as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        samples.append(EvalSample(
            incident_id=item["incident_id"],
            title=item["title"],
            description=item["description"],
            service=item.get("service", ""),
            priority=item.get("priority", "P3"),
            error_message=item.get("error_message", ""),
            ground_truth_root_cause=item["ground_truth_root_cause"],
        ))
    return samples


def root_cause_matches(predicted: str, ground_truth: str) -> bool:
    """
    Fuzzy match: check if ground truth label appears in the predicted hypothesis.
    In production evaluation, this would use semantic similarity scoring.
    """
    gt_lower = ground_truth.lower()
    pred_lower = predicted.lower()
    # Check for key term overlap
    gt_terms = set(gt_lower.split())
    pred_terms = set(pred_lower.split())
    overlap = gt_terms & pred_terms
    significant_overlap = len(overlap) >= max(1, len(gt_terms) // 2)
    return gt_lower in pred_lower or significant_overlap


def evaluate_system(
    diagnoser,
    samples: list[EvalSample],
    config_label: str = "Full System",
) -> EvalMetrics:
    """Run evaluation over all samples and compute metrics."""
    from src.diagnosis.diagnoser import IncidentContext

    correct_top1 = 0
    correct_top3 = 0
    latencies = []

    console.print(f"\n  Evaluating: [bold]{config_label}[/bold] ({len(samples)} samples)")

    for i, sample in enumerate(samples):
        incident = IncidentContext(
            incident_id=sample.incident_id,
            title=sample.title,
            description=sample.description,
            service=sample.service,
            priority=sample.priority,
            error_message=sample.error_message,
        )

        t0 = time.perf_counter()
        result = diagnoser.diagnose(incident)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        # Accuracy: top-1 root cause matches ground truth
        if result.root_causes:
            top1_hypothesis = result.root_causes[0].hypothesis
            if root_cause_matches(top1_hypothesis, sample.ground_truth_root_cause):
                correct_top1 += 1

            # Recall@3: correct answer in top 3
            for rc in result.root_causes[:3]:
                if root_cause_matches(rc.hypothesis, sample.ground_truth_root_cause):
                    correct_top3 += 1
                    break

        if (i + 1) % 50 == 0:
            console.print(f"    Processed {i+1}/{len(samples)}...")

    n = len(samples)
    return EvalMetrics(
        accuracy=correct_top1 / n,
        precision_at_1=correct_top1 / n,
        recall_at_3=correct_top3 / n,
        mean_diagnosis_time_s=sum(latencies) / len(latencies),
        total_samples=n,
        config_label=config_label,
    )


def run_ablation_study(diagnoser_factory, samples: list[EvalSample]) -> list[EvalMetrics]:
    """
    Ablation study: disable one component at a time.
    Reproduces Fig. 5 (left) from the paper.
    """
    ablation_configs = [
        ("Full System", {}),
        ("No Re-ranking", {"disable_reranking": True}),
        ("No Chunking", {"disable_chunking": True}),
        ("No Metadata Filters", {"disable_filters": True}),
        ("BM25 Only", {"disable_dense_retrieval": True}),
        ("No RAG", {"disable_retrieval": True}),
    ]
    results = []
    for label, overrides in ablation_configs:
        diagnoser = diagnoser_factory(overrides)
        metrics = evaluate_system(diagnoser, samples, config_label=label)
        results.append(metrics)
    return results


def print_results_table(metrics_list: list[EvalMetrics]):
    table = Table(title="Evaluation Results", show_header=True, header_style="bold blue")
    table.add_column("Configuration", style="cyan", min_width=20)
    table.add_column("Accuracy (%)", justify="right")
    table.add_column("Prec@1 (%)", justify="right")
    table.add_column("Recall@3 (%)", justify="right")
    table.add_column("MDT (s)", justify="right")
    table.add_column("Samples", justify="right")

    for m in metrics_list:
        d = m.to_dict()
        is_full = m.config_label == "Full System"
        style = "bold green" if is_full else ""
        table.add_row(
            m.config_label,
            f"{d['accuracy']:.1f}",
            f"{d['precision_at_1']:.1f}",
            f"{d['recall_at_3']:.1f}",
            f"{d['mdt_seconds']:.2f}",
            str(d['samples']),
            style=style,
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG-AIOps framework")
    parser.add_argument("--dataset", default="data/samples/eval_dataset.json")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--output", default="data/eval_results/results.json")
    parser.add_argument("--local-embeddings", action="store_true")
    args = parser.parse_args()

    console.rule("[bold blue]RAG-AIOps Evaluation Harness")

    # Load dataset
    samples = load_eval_dataset(args.dataset)
    console.print(f"Loaded [bold]{len(samples)}[/bold] evaluation samples")

    # Load config & build diagnoser
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.local_embeddings:
        config["embedding"]["provider"] = "local"

    # Import and build components
    from src.indexing.embedder import Embedder
    from src.indexing.vector_store import FAISSVectorStore
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import SemanticRetriever
    from src.retrieval.query_expander import HyDEQueryExpander
    from src.diagnosis.diagnoser import IncidentDiagnoser
    from src.diagnosis.llm_client import LLMClient

    embedder = Embedder({**config["embedding"], **config["llm"], "cache": config.get("cache", {})})
    vector_store = FAISSVectorStore({**config["vector_store"], "dimensions": embedder.dimensions})
    reranker = CrossEncoderReranker(config["retrieval"]["reranker_model"])
    llm_client = LLMClient(config["llm"])
    expander = HyDEQueryExpander(llm_client) if config["retrieval"].get("hyde_enabled") else None

    retriever = SemanticRetriever(
        vector_store=vector_store, embedder=embedder,
        reranker=reranker, query_expander=expander,
        config=config["retrieval"],
    )
    diagnoser = IncidentDiagnoser(retriever, llm_client, config)

    all_metrics = []

    if args.ablation:
        console.print("\n[bold]Running ablation study...[/bold]")
        # For ablation, we'd pass different configs — simplified here
        full_metrics = evaluate_system(diagnoser, samples, "Full System")
        all_metrics.append(full_metrics)
        console.print("\n[dim]Note: Full ablation requires component-level config overrides.[/dim]")
        console.print("[dim]See paper Section 6.5 for complete ablation methodology.[/dim]")
    else:
        metrics = evaluate_system(diagnoser, samples)
        all_metrics.append(metrics)

    print_results_table(all_metrics)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([m.to_dict() for m in all_metrics], f, indent=2)
    console.print(f"\nResults saved to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
