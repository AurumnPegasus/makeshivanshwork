"""
Retrieval Metrics for Memory Architecture Research

Implements standard IR metrics:
- Recall@k: Fraction of relevant items retrieved
- Precision@k: Fraction of retrieved items that are relevant
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

All metrics designed for efficiency:
- Vectorized computations where possible
- Early termination
- No unnecessary allocations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class RetrievalMetrics:
    """Container for all retrieval metrics"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0

    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0  # Normalized DCG

    # Additional
    num_queries: int = 0
    num_relevant: int = 0
    avg_retrieved: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "num_queries": self.num_queries,
        }


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Compute Recall@k.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: List of retrieved item IDs (ordered by relevance)
        relevant_ids: Set of relevant item IDs
        k: Number of top results to consider

    Returns:
        Recall@k score (0-1)
    """
    if not relevant_ids:
        return 1.0  # If nothing is relevant, recall is perfect

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Compute Precision@k.

    Precision@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved_ids: List of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k: Number of top results to consider

    Returns:
        Precision@k score (0-1)
    """
    if k == 0:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    hits = len(retrieved_set & relevant_set)
    return hits / min(k, len(retrieved_ids)) if retrieved_ids else 0.0


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = 1 / rank_of_first_relevant

    Args:
        retrieved_ids: List of retrieved item IDs
        relevant_ids: Set of relevant item IDs

    Returns:
        Reciprocal rank (0-1)
    """
    relevant_set = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    for i, item_id in enumerate(retrieved_ids):
        if item_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg(
    relevance_scores: List[float],
    k: Optional[int] = None,
) -> float:
    """
    Compute Discounted Cumulative Gain.

    DCG@k = Σ (2^rel_i - 1) / log2(i + 2)

    Args:
        relevance_scores: List of relevance scores (ordered)
        k: Number of positions to consider

    Returns:
        DCG score
    """
    if not relevance_scores:
        return 0.0

    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += (2 ** rel - 1) / np.log2(i + 2)

    return dcg


def compute_ndcg(
    retrieved_ids: List[str],
    relevance_map: Dict[str, float],
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.

    NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: List of retrieved item IDs
        relevance_map: Map from item ID to relevance score
        k: Number of positions to consider

    Returns:
        NDCG@k score (0-1)
    """
    # Get relevance scores for retrieved items
    retrieved_scores = [
        relevance_map.get(item_id, 0.0)
        for item_id in retrieved_ids[:k]
    ]

    # Compute DCG
    dcg = compute_dcg(retrieved_scores, k)

    # Compute ideal DCG (sorted relevance scores)
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = compute_dcg(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_all_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    relevance_map: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_ids: List of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        relevance_map: Optional graded relevance (for NDCG)

    Returns:
        Dictionary of metric names to scores
    """
    # Binary relevance map if not provided
    if relevance_map is None:
        relevance_map = {item_id: 1.0 for item_id in relevant_ids}

    metrics = {
        "recall@1": compute_recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall@3": compute_recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall@5": compute_recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall@10": compute_recall_at_k(retrieved_ids, relevant_ids, 10),
        "recall@20": compute_recall_at_k(retrieved_ids, relevant_ids, 20),
        "precision@1": compute_precision_at_k(retrieved_ids, relevant_ids, 1),
        "precision@3": compute_precision_at_k(retrieved_ids, relevant_ids, 3),
        "precision@5": compute_precision_at_k(retrieved_ids, relevant_ids, 5),
        "precision@10": compute_precision_at_k(retrieved_ids, relevant_ids, 10),
        "mrr": compute_mrr(retrieved_ids, relevant_ids),
        "ndcg@10": compute_ndcg(retrieved_ids, relevance_map, 10),
    }

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
) -> RetrievalMetrics:
    """
    Aggregate metrics across multiple queries.

    Args:
        all_metrics: List of metric dictionaries

    Returns:
        Aggregated RetrievalMetrics
    """
    if not all_metrics:
        return RetrievalMetrics()

    # Average each metric
    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        agg[key] = np.mean(values) if values else 0.0

    return RetrievalMetrics(
        recall_at_1=agg.get("recall@1", 0.0),
        recall_at_3=agg.get("recall@3", 0.0),
        recall_at_5=agg.get("recall@5", 0.0),
        recall_at_10=agg.get("recall@10", 0.0),
        recall_at_20=agg.get("recall@20", 0.0),
        precision_at_1=agg.get("precision@1", 0.0),
        precision_at_3=agg.get("precision@3", 0.0),
        precision_at_5=agg.get("precision@5", 0.0),
        precision_at_10=agg.get("precision@10", 0.0),
        mrr=agg.get("mrr", 0.0),
        ndcg_at_10=agg.get("ndcg@10", 0.0),
        num_queries=len(all_metrics),
    )


class MetricsTracker:
    """
    Track metrics across evaluation.

    Provides running statistics without storing all results.
    """

    def __init__(self):
        self._counts = {}
        self._sums = {}
        self._num_samples = 0

    def add(self, metrics: Dict[str, float]):
        """Add a sample of metrics"""
        self._num_samples += 1
        for key, value in metrics.items():
            if key not in self._counts:
                self._counts[key] = 0
                self._sums[key] = 0.0
            self._counts[key] += 1
            self._sums[key] += value

    def get_averages(self) -> Dict[str, float]:
        """Get average of all tracked metrics"""
        return {
            key: self._sums[key] / self._counts[key]
            for key in self._sums
            if self._counts[key] > 0
        }

    def get_count(self) -> int:
        """Get number of samples"""
        return self._num_samples
