"""
Ablation Study Framework

Systematically tests contribution of each component:
- Multi-embedding vs single embedding
- With/without entity extraction
- With/without intent classification
- With/without graph structure
- Different embedding dimensions (MRL)

METHODOLOGY:
1. Start with full system
2. Remove one component at a time
3. Measure performance drop
4. Identify critical components
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy


@dataclass
class AblationConfig:
    """Configuration for ablation experiment"""
    name: str
    description: str
    modifications: Dict[str, Any]  # Config changes to make
    expected_impact: str  # High/Medium/Low


@dataclass
class AblationResult:
    """Result of a single ablation"""
    config: AblationConfig
    baseline_metrics: Dict[str, float]
    ablated_metrics: Dict[str, float]
    delta: Dict[str, float]  # Difference from baseline
    relative_delta: Dict[str, float]  # Percentage change


STANDARD_ABLATIONS = [
    AblationConfig(
        name="no_multi_embedding",
        description="Use single content embedding only",
        modifications={"use_multi_embedding": False},
        expected_impact="High",
    ),
    AblationConfig(
        name="no_entity_extraction",
        description="Disable entity extraction",
        modifications={"entity_extractor": None},
        expected_impact="Medium",
    ),
    AblationConfig(
        name="no_intent_classification",
        description="Disable intent classification",
        modifications={"intent_classifier": None},
        expected_impact="High",
    ),
    AblationConfig(
        name="no_graph",
        description="Disable graph structure (vector only)",
        modifications={"use_graph": False},
        expected_impact="Medium",
    ),
    AblationConfig(
        name="no_hypergraph",
        description="Use pairwise graph instead of hypergraph",
        modifications={"use_hypergraph": False},
        expected_impact="Medium",
    ),
    AblationConfig(
        name="no_hierarchy",
        description="Flat storage (no working/episodic/semantic)",
        modifications={"use_hierarchy": False},
        expected_impact="Medium",
    ),
    AblationConfig(
        name="small_embeddings",
        description="Use 256d instead of 768d embeddings (MRL)",
        modifications={"embedding_dimensions": 256},
        expected_impact="Low",
    ),
    AblationConfig(
        name="no_temporal_decay",
        description="No recency weighting",
        modifications={"use_temporal_decay": False},
        expected_impact="Low",
    ),
    AblationConfig(
        name="no_caching",
        description="Disable all caching",
        modifications={"cache_enabled": False},
        expected_impact="Low",
    ),
]


def run_ablation_study(
    system_factory: Callable,
    evaluator: Any,
    messages: List[Dict],
    test_cases: List[Dict],
    ablations: Optional[List[AblationConfig]] = None,
) -> List[AblationResult]:
    """
    Run ablation study on a memory system.

    Args:
        system_factory: Callable that creates system from config
        evaluator: Evaluator instance
        messages: Seed messages
        test_cases: Test cases
        ablations: Ablation configs (default: STANDARD_ABLATIONS)

    Returns:
        List of AblationResults
    """
    ablations = ablations or STANDARD_ABLATIONS
    results = []

    # Run baseline
    print("Running baseline...")
    baseline_system = system_factory({})
    baseline_eval = evaluator.evaluate_system(baseline_system, messages, test_cases)
    baseline_metrics = _extract_key_metrics(baseline_eval)

    # Run each ablation
    for ablation in ablations:
        print(f"Running ablation: {ablation.name}...")

        try:
            # Create ablated system
            ablated_system = system_factory(ablation.modifications)

            # Evaluate
            ablated_eval = evaluator.evaluate_system(ablated_system, messages, test_cases)
            ablated_metrics = _extract_key_metrics(ablated_eval)

            # Calculate deltas
            delta = {}
            relative_delta = {}
            for key in baseline_metrics:
                delta[key] = ablated_metrics.get(key, 0) - baseline_metrics[key]
                if baseline_metrics[key] != 0:
                    relative_delta[key] = (delta[key] / baseline_metrics[key]) * 100
                else:
                    relative_delta[key] = 0

            results.append(AblationResult(
                config=ablation,
                baseline_metrics=baseline_metrics,
                ablated_metrics=ablated_metrics,
                delta=delta,
                relative_delta=relative_delta,
            ))

        except Exception as e:
            print(f"  Ablation {ablation.name} failed: {e}")
            continue

    return results


def _extract_key_metrics(eval_result) -> Dict[str, float]:
    """Extract key metrics from evaluation result"""
    metrics = {}

    # Retrieval metrics
    if hasattr(eval_result, "retrieval"):
        metrics["recall@5"] = eval_result.retrieval.recall_at_5
        metrics["mrr"] = eval_result.retrieval.mrr
        metrics["precision@5"] = eval_result.retrieval.precision_at_5

    # Answer quality
    if hasattr(eval_result, "answer_quality") and eval_result.answer_quality:
        metrics["answer_overall"] = eval_result.answer_quality.get("overall", 0)

    # Safety
    metrics["false_action_rate"] = getattr(eval_result, "false_action_rate", 0)

    # Efficiency
    metrics["latency_p50"] = getattr(eval_result, "latency_p50_ms", 0)

    return metrics


def format_ablation_results(
    results: List[AblationResult],
    format: str = "markdown",
) -> str:
    """
    Format ablation results as table.

    Args:
        results: Ablation results
        format: Output format (markdown, latex)

    Returns:
        Formatted table
    """
    if not results:
        return "No results"

    headers = ["Component Removed", "R@5 Δ", "MRR Δ", "FAR Δ", "Impact"]
    rows = []

    for r in results:
        row = [
            r.config.name,
            f"{r.relative_delta.get('recall@5', 0):+.1f}%",
            f"{r.relative_delta.get('mrr', 0):+.1f}%",
            f"{r.relative_delta.get('false_action_rate', 0):+.1f}%",
            r.config.expected_impact,
        ]
        rows.append(row)

    # Sort by impact (most negative first)
    rows.sort(key=lambda x: float(x[1].replace("%", "").replace("+", "")))

    if format == "markdown":
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    elif format == "latex":
        lines = []
        lines.append(r"\begin{tabular}{" + "l" * len(headers) + "}")
        lines.append(r"\toprule")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")
        for row in rows:
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    return str(rows)


def identify_critical_components(
    results: List[AblationResult],
    threshold: float = -5.0,  # Percentage drop
) -> List[str]:
    """
    Identify components whose removal causes significant performance drop.

    Args:
        results: Ablation results
        threshold: Minimum % drop to be considered critical

    Returns:
        List of critical component names
    """
    critical = []

    for r in results:
        # Check if recall@5 drops significantly
        if r.relative_delta.get("recall@5", 0) <= threshold:
            critical.append(r.config.name)
        # Or if false action rate increases significantly
        elif r.relative_delta.get("false_action_rate", 0) >= abs(threshold):
            critical.append(r.config.name)

    return critical
