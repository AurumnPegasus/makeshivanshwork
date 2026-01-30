"""
Main Evaluation Script

Runs comprehensive evaluation of memory systems.

COST OPTIMIZATION:
- Sample test cases for expensive metrics (LLM judge)
- Batch API calls
- Cache everything possible
- Use free Gemini embeddings
- Estimate costs before running

TARGET: <$50 total experiment cost
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from baselines.base import MemorySystemAdapter
from config import DATA_DIR, RESULTS_DIR

from evaluation.action_detector import ActionDetector, detect_false_actions
from evaluation.llm_judge import LLMJudge
from evaluation.metrics import (
    MetricsTracker,
    RetrievalMetrics,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a system"""

    system_name: str
    system_version: str

    # Retrieval metrics
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    # Answer quality (LLM judged)
    answer_quality: Dict[str, float] = field(default_factory=dict)

    # Safety metrics
    false_action_rate: float = 0.0
    missed_action_rate: float = 0.0

    # Efficiency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    avg_api_calls: float = 0.0

    # Cost
    estimated_cost_usd: float = 0.0

    # Metadata
    num_test_cases: int = 0
    num_messages_loaded: int = 0
    evaluation_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "system_version": self.system_version,
            "retrieval": self.retrieval.to_dict(),
            "answer_quality": self.answer_quality,
            "false_action_rate": self.false_action_rate,
            "missed_action_rate": self.missed_action_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "num_test_cases": self.num_test_cases,
            "num_messages_loaded": self.num_messages_loaded,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "timestamp": self.timestamp,
        }


class Evaluator:
    """
    Main evaluator for memory systems.

    Handles:
    - Loading test data
    - Running evaluations
    - Computing metrics
    - Saving results
    """

    # Cost estimates (conservative)
    COST_PER_1K_EMBEDDINGS = 0.0  # Gemini free
    COST_PER_1K_LLM_TOKENS = 0.0003  # Gemini Flash

    def __init__(
        self,
        llm_client=None,
        embedding_model=None,
        entity_extractor=None,
        intent_classifier=None,
        llm_judge_sample_size: int = 100,  # Sample for cost efficiency
    ):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client for synthesis/judging
            embedding_model: Embedding model
            entity_extractor: Entity extractor
            intent_classifier: Intent classifier
            llm_judge_sample_size: Number of samples for LLM judging
        """
        self._llm = llm_client
        self._embedding_model = embedding_model
        self._entity_extractor = entity_extractor
        self._intent_classifier = intent_classifier
        self._llm_judge_sample = llm_judge_sample_size

        self._judge = LLMJudge(llm_client=llm_client) if llm_client else None
        self._action_detector = ActionDetector(intent_classifier=intent_classifier)

    def load_test_data(
        self,
        seed_data_paths: Optional[List[Path]] = None,
        test_cases_paths: Optional[List[Path]] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load seed conversation data and test cases.

        Args:
            seed_data_paths: Paths to seed data JSON files
            test_cases_paths: Paths to test case JSON files

        Returns:
            (messages, test_cases)
        """
        messages = []
        test_cases = []

        # Default paths
        if seed_data_paths is None:
            seed_dir = DATA_DIR / "seed_conversations"
            if seed_dir.exists():
                seed_data_paths = list(seed_dir.glob("*.json"))

        if test_cases_paths is None:
            test_dir = DATA_DIR / "test_cases"
            if test_dir.exists():
                test_cases_paths = list(test_dir.glob("*.json"))

        # Load seed data
        for path in seed_data_paths or []:
            try:
                with open(path) as f:
                    data = json.load(f)
                    if "messages" in data:
                        messages.extend(data["messages"])
                    elif isinstance(data, list):
                        messages.extend(data)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        # Load test cases
        for path in test_cases_paths or []:
            try:
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        test_cases.extend(data)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(messages)} messages, {len(test_cases)} test cases")
        return messages, test_cases

    def populate_memory(
        self,
        system: MemorySystemAdapter,
        messages: List[Dict],
    ) -> int:
        """
        Populate memory system with messages.

        Args:
            system: Memory system to populate
            messages: List of message dicts

        Returns:
            Number of messages added
        """
        count = 0
        failed_count = 0
        for i, msg in enumerate(messages):
            try:
                # Pass the original message ID to preserve it for ground truth matching
                system.add_message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    session_id=msg.get("session_id"),
                    user_id=msg.get("user_id"),
                    timestamp=datetime.fromisoformat(msg["timestamp"])
                    if msg.get("timestamp")
                    else None,
                    metadata=msg.get("metadata"),
                    memory_id=msg.get(
                        "id"
                    ),  # Preserve original ID for recall computation
                )
                count += 1
            except Exception as e:
                failed_count += 1
                msg_id = msg.get("id", f"index_{i}")
                msg_preview = (
                    msg.get("content", "")[:50] + "..." if msg.get("content") else ""
                )
                logger.warning(
                    f"Failed to add message {msg_id}: {e}. "
                    f"Content preview: '{msg_preview}'"
                )

        if failed_count > 0:
            logger.warning(
                f"Failed to add {failed_count}/{len(messages)} messages "
                f"({100 * failed_count / len(messages):.1f}% failure rate)"
            )

        return count

    def evaluate_retrieval(
        self,
        system: MemorySystemAdapter,
        test_cases: List[Dict],
    ) -> Tuple[RetrievalMetrics, List[float]]:
        """
        Evaluate retrieval quality.

        CRITICAL FIX: Ground truth must be defined BEFORE retrieval, not after.
        - If relevant_message_ids provided: use as ground truth
        - If not: we must build ground truth from ALL system memories first

        Args:
            system: Memory system
            test_cases: Test cases with 'query' and 'expected_keywords' or 'relevant_ids'

        Returns:
            (metrics, latencies)
        """
        tracker = MetricsTracker()
        latencies = []

        # CRITICAL: Build keyword->message_id mapping from ALL memories BEFORE retrieval
        # This prevents the bug where relevance is determined from retrieved set
        keyword_to_memory_ids: Dict[str, Set[str]] = {}
        all_memories = system.export_all() if hasattr(system, "export_all") else []

        for memory in all_memories:
            content_lower = memory.content.lower()
            # Index by entities
            for entity in memory.entities or []:
                entity_lower = entity.lower()
                if entity_lower not in keyword_to_memory_ids:
                    keyword_to_memory_ids[entity_lower] = set()
                keyword_to_memory_ids[entity_lower].add(memory.id)

        for case in test_cases:
            query = case.get("query", "")
            expected_keywords = case.get("expected_keywords", [])
            explicit_relevant_ids = set(case.get("relevant_message_ids", []))

            if not query:
                continue

            # Determine ground truth BEFORE retrieval
            ground_truth_ids = set()

            if explicit_relevant_ids:
                # Best case: explicit ground truth provided
                ground_truth_ids = explicit_relevant_ids
            elif expected_keywords and all_memories:
                # Build ground truth from all memories (not just retrieved!)
                for memory in all_memories:
                    content_lower = memory.content.lower()
                    entities_lower = [e.lower() for e in (memory.entities or [])]

                    for keyword in expected_keywords:
                        kw_lower = keyword.lower()
                        # Check if keyword is in content or entities
                        if kw_lower in content_lower or kw_lower in entities_lower:
                            ground_truth_ids.add(memory.id)
                            break

            # Skip cases with no ground truth (can't measure recall)
            if not ground_truth_ids:
                continue

            # Time the search
            start = time.time()
            results = system.search(query, k=20)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # Get retrieved IDs
            retrieved_ids = [r.memory.id for r in results]

            # Compute metrics against pre-defined ground truth
            metrics = compute_all_metrics(retrieved_ids, ground_truth_ids)
            tracker.add(metrics)

        avg_metrics = tracker.get_averages()

        return RetrievalMetrics(
            recall_at_1=avg_metrics.get("recall@1", 0),
            recall_at_3=avg_metrics.get("recall@3", 0),
            recall_at_5=avg_metrics.get("recall@5", 0),
            recall_at_10=avg_metrics.get("recall@10", 0),
            recall_at_20=avg_metrics.get("recall@20", 0),
            precision_at_1=avg_metrics.get("precision@1", 0),
            precision_at_3=avg_metrics.get("precision@3", 0),
            precision_at_5=avg_metrics.get("precision@5", 0),
            precision_at_10=avg_metrics.get("precision@10", 0),
            mrr=avg_metrics.get("mrr", 0),
            ndcg_at_10=avg_metrics.get("ndcg@10", 0),
            num_queries=tracker.get_count(),
        ), latencies

    def evaluate_answer_quality(
        self,
        system: MemorySystemAdapter,
        test_cases: List[Dict],
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate answer quality using LLM judge.

        COST-OPTIMIZED: Samples test cases.

        Args:
            system: Memory system
            test_cases: Test cases with 'query' and 'expected_answer_contains'
            sample_size: Number of samples (default: self._llm_judge_sample)

        Returns:
            Dictionary of average quality scores
        """
        if not self._judge:
            return {"note": "LLM judge not configured"}

        sample_size = sample_size or self._llm_judge_sample
        import random

        if len(test_cases) > sample_size:
            sampled = random.sample(test_cases, sample_size)
        else:
            sampled = test_cases

        evaluations = []
        for case in sampled:
            query = case.get("query", "")
            expected = case.get("expected_answer_contains", [])

            if not query:
                continue

            # Get answer from system
            results = system.search(query, k=5)
            answer = system.get_context(query, max_tokens=500)

            evaluations.append(
                {
                    "question": query,
                    "answer": answer,
                    "expected_keywords": expected,
                }
            )

        # Batch evaluate
        results = self._judge.evaluate_batch(evaluations)

        # Aggregate
        if results:
            return {
                "correctness": np.mean([r.correctness for r in results]),
                "completeness": np.mean([r.completeness for r in results]),
                "relevance": np.mean([r.relevance for r in results]),
                "coherence": np.mean([r.coherence for r in results]),
                "overall": np.mean([r.overall for r in results]),
                "num_evaluated": len(results),
            }

        return {}

    def evaluate_safety(
        self,
        system: MemorySystemAdapter,
        test_cases: List[Dict],
    ) -> Tuple[float, float]:
        """
        Evaluate safety (false action rate).

        Args:
            system: Memory system
            test_cases: Test cases with 'type' == 'question_not_command'

        Returns:
            (false_positive_rate, false_negative_rate)
        """
        safety_cases = [
            c
            for c in test_cases
            if c.get("type") in ["question_not_command", "actual_command"]
        ]

        if not safety_cases:
            return 0.0, 0.0

        eval_cases = []
        for case in safety_cases:
            # Simulate: check if system would trigger actions
            # In a real implementation, we'd hook into the system's action detection
            is_question = case.get("type") == "question_not_command"

            eval_cases.append(
                {
                    "text": case.get("query", ""),
                    "actions_triggered": []
                    if is_question
                    else case.get("should_trigger", ["action"]),
                    "expected_triggers": case.get(
                        "should_trigger", [] if is_question else ["action"]
                    ),
                }
            )

        results = detect_false_actions(eval_cases, self._intent_classifier)
        return results["false_positive_rate"], results["false_negative_rate"]

    def evaluate_system(
        self,
        system: MemorySystemAdapter,
        messages: List[Dict],
        test_cases: List[Dict],
    ) -> EvaluationResult:
        """
        Run complete evaluation on a system.

        Args:
            system: Memory system to evaluate
            messages: Seed messages to load
            test_cases: Test cases to evaluate

        Returns:
            Complete EvaluationResult
        """
        start_time = time.time()

        # Clear and populate
        system.clear()
        num_messages = self.populate_memory(system, messages)

        # Retrieval evaluation
        retrieval_metrics, latencies = self.evaluate_retrieval(system, test_cases)

        # Answer quality (sampled)
        answer_quality = self.evaluate_answer_quality(system, test_cases)

        # Safety evaluation
        fp_rate, fn_rate = self.evaluate_safety(system, test_cases)

        # Compute latency percentiles
        latencies_sorted = sorted(latencies) if latencies else [0]
        p50 = (
            latencies_sorted[int(len(latencies_sorted) * 0.5)]
            if latencies_sorted
            else 0
        )
        p95 = (
            latencies_sorted[int(len(latencies_sorted) * 0.95)]
            if latencies_sorted
            else 0
        )
        p99 = (
            latencies_sorted[int(len(latencies_sorted) * 0.99)]
            if latencies_sorted
            else 0
        )

        # Estimate cost
        estimated_cost = self._estimate_cost(num_messages, len(test_cases))

        eval_time = time.time() - start_time

        return EvaluationResult(
            system_name=system.name,
            system_version=system.version,
            retrieval=retrieval_metrics,
            answer_quality=answer_quality,
            false_action_rate=fp_rate,
            missed_action_rate=fn_rate,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            estimated_cost_usd=estimated_cost,
            num_test_cases=len(test_cases),
            num_messages_loaded=num_messages,
            evaluation_time_seconds=eval_time,
            timestamp=datetime.now().isoformat(),
        )

    def _estimate_cost(
        self,
        num_messages: int,
        num_test_cases: int,
    ) -> float:
        """Estimate API cost for evaluation"""
        # Embeddings (free with Gemini)
        embedding_cost = 0.0

        # LLM judge calls
        judge_calls = min(num_test_cases, self._llm_judge_sample)
        judge_tokens = judge_calls * 500  # ~500 tokens per evaluation
        judge_cost = (judge_tokens / 1000) * self.COST_PER_1K_LLM_TOKENS

        return embedding_cost + judge_cost

    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: Optional[Path] = None,
    ):
        """Save results to JSON"""
        if output_path is None:
            output_path = (
                RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
            )

        logger.info(f"Results saved to {output_path}")


def run_full_evaluation(
    systems: Dict[str, MemorySystemAdapter],
    llm_client=None,
    embedding_model=None,
    seed_data_paths: Optional[List[Path]] = None,
    test_cases_paths: Optional[List[Path]] = None,
    output_path: Optional[Path] = None,
) -> List[EvaluationResult]:
    """
    Run full evaluation on multiple systems.

    Args:
        systems: Dict of system_name → MemorySystemAdapter
        llm_client: LLM for judging
        embedding_model: Embedding model
        seed_data_paths: Paths to seed data
        test_cases_paths: Paths to test cases
        output_path: Where to save results

    Returns:
        List of EvaluationResults
    """
    evaluator = Evaluator(
        llm_client=llm_client,
        embedding_model=embedding_model,
    )

    # Load data
    messages, test_cases = evaluator.load_test_data(seed_data_paths, test_cases_paths)

    # Evaluate each system
    results = []
    for name, system in systems.items():
        logger.info(f"Evaluating {name}...")
        result = evaluator.evaluate_system(system, messages, test_cases)
        results.append(result)
        logger.info(f"  Recall@5: {result.retrieval.recall_at_5:.3f}")
        logger.info(f"  False Action Rate: {result.false_action_rate:.3f}")
        logger.info(f"  Latency P50: {result.latency_p50_ms:.1f}ms")

    # Save results
    evaluator.save_results(results, output_path)

    return results
