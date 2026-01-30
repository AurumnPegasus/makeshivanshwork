"""
Evaluation Framework for Memory Architecture Research

Implements comprehensive, cost-efficient evaluation:

METRICS:
1. Retrieval Quality
   - Recall@k (k=1,3,5,10,20)
   - Precision@k
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG)

2. Answer Quality (LLM-judged)
   - Correctness (1-5)
   - Completeness (1-5)
   - Relevance (1-5)
   - Coherence (1-5)

3. Safety/Robustness
   - False Action Rate
   - Hallucination Rate
   - Contradiction Detection

4. Efficiency
   - Latency (P50, P95, P99)
   - Memory Usage
   - API Costs

COST OPTIMIZATION:
- Batch LLM judge calls
- Cache embeddings
- Sample for expensive metrics
- Use free Gemini API
"""

import sys
from pathlib import Path

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg,
    RetrievalMetrics,
)
from evaluation.llm_judge import (
    LLMJudge,
    JudgeResult,
    batch_judge,
)
from evaluation.action_detector import (
    ActionDetector,
    detect_false_actions,
)
from evaluation.run_eval import (
    Evaluator,
    EvaluationResult,
    run_full_evaluation,
)

__all__ = [
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_ndcg",
    "RetrievalMetrics",
    "LLMJudge",
    "JudgeResult",
    "batch_judge",
    "ActionDetector",
    "detect_false_actions",
    "Evaluator",
    "EvaluationResult",
    "run_full_evaluation",
]
