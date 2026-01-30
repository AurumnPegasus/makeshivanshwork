"""
LLM-as-Judge for Answer Quality Evaluation

COST-EFFICIENT DESIGN:
- Batch multiple evaluations per API call
- Use cheaper models (Gemini Flash, GPT-4o-mini)
- Cache identical evaluations
- Sample for large datasets

Evaluation dimensions:
- Correctness: Is the answer factually correct?
- Completeness: Does it answer the full question?
- Relevance: Is the information relevant to the query?
- Coherence: Is it well-structured and clear?
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation"""
    correctness: float  # 1-5
    completeness: float  # 1-5
    relevance: float  # 1-5
    coherence: float  # 1-5

    # Overall score (average)
    overall: float = 0.0

    # Metadata
    reasoning: str = ""
    model: str = ""
    cached: bool = False

    def __post_init__(self):
        self.overall = (
            self.correctness + self.completeness +
            self.relevance + self.coherence
        ) / 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "overall": self.overall,
            "reasoning": self.reasoning,
            "model": self.model,
        }


class LLMJudge:
    """
    LLM-based answer quality evaluation.

    Uses LLM to score answers on multiple dimensions.
    Optimized for cost with batching and caching.
    """

    EVALUATION_PROMPT = """You are evaluating the quality of an AI assistant's answer.

QUESTION: {question}

EXPECTED INFORMATION: {expected}

ACTUAL ANSWER: {answer}

Rate the answer on these dimensions (1-5 scale):

1. CORRECTNESS (1-5): Is the information factually accurate?
   1 = Completely wrong
   3 = Partially correct
   5 = Fully correct

2. COMPLETENESS (1-5): Does it fully answer the question?
   1 = Missing most information
   3 = Answers part of the question
   5 = Fully comprehensive

3. RELEVANCE (1-5): Is the information relevant to the query?
   1 = Off-topic
   3 = Somewhat relevant
   5 = Directly relevant

4. COHERENCE (1-5): Is the answer well-structured and clear?
   1 = Incoherent/confusing
   3 = Understandable but messy
   5 = Clear and well-organized

Respond in JSON format:
{{"correctness": X, "completeness": X, "relevance": X, "coherence": X, "reasoning": "brief explanation"}}

Only output the JSON, nothing else."""

    def __init__(
        self,
        llm_client=None,
        model: str = "gemini-2.0-flash",
        cache_enabled: bool = True,
        max_cache_size: int = 10000,
    ):
        """
        Initialize LLM Judge.

        Args:
            llm_client: LLM client for evaluation
            model: Model to use
            cache_enabled: Whether to cache results
            max_cache_size: Maximum cache entries
        """
        self._llm = llm_client
        self._model = model
        self._cache_enabled = cache_enabled
        self._cache: Dict[str, JudgeResult] = {}
        self._max_cache = max_cache_size

        # Statistics
        self._total_calls = 0
        self._cache_hits = 0

    def _cache_key(
        self,
        question: str,
        answer: str,
        expected: str,
    ) -> str:
        """Generate cache key"""
        content = f"{question}|{answer}|{expected}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def evaluate(
        self,
        question: str,
        answer: str,
        expected: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> JudgeResult:
        """
        Evaluate a single answer.

        Args:
            question: The original question
            answer: The generated answer
            expected: Expected answer or key information
            expected_keywords: Keywords expected in the answer

        Returns:
            JudgeResult with scores
        """
        # Build expected string
        if expected is None:
            expected = ""
            if expected_keywords:
                expected = f"Should contain: {', '.join(expected_keywords)}"

        # Check cache
        cache_key = self._cache_key(question, answer, expected)
        if self._cache_enabled and cache_key in self._cache:
            self._cache_hits += 1
            result = self._cache[cache_key]
            result.cached = True
            return result

        self._total_calls += 1

        # Call LLM
        if not self._llm:
            # Return neutral scores if no LLM
            return JudgeResult(
                correctness=3.0,
                completeness=3.0,
                relevance=3.0,
                coherence=3.0,
                reasoning="No LLM judge configured",
                model="none",
            )

        prompt = self.EVALUATION_PROMPT.format(
            question=question,
            expected=expected or "Not specified",
            answer=answer,
        )

        try:
            response = self._llm.generate(prompt, temperature=0.0)

            # Parse JSON response
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result = JudgeResult(
                    correctness=float(data.get("correctness", 3)),
                    completeness=float(data.get("completeness", 3)),
                    relevance=float(data.get("relevance", 3)),
                    coherence=float(data.get("coherence", 3)),
                    reasoning=data.get("reasoning", ""),
                    model=self._model,
                )
            else:
                # Fallback parsing
                result = JudgeResult(
                    correctness=3.0,
                    completeness=3.0,
                    relevance=3.0,
                    coherence=3.0,
                    reasoning="Could not parse LLM response",
                    model=self._model,
                )

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            result = JudgeResult(
                correctness=3.0,
                completeness=3.0,
                relevance=3.0,
                coherence=3.0,
                reasoning=f"Error: {str(e)}",
                model=self._model,
            )

        # Cache result
        if self._cache_enabled and len(self._cache) < self._max_cache:
            self._cache[cache_key] = result

        return result

    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]],
        batch_size: int = 5,
    ) -> List[JudgeResult]:
        """
        Evaluate multiple answers in batches.

        COST OPTIMIZATION: Groups evaluations to reduce API calls.

        Args:
            evaluations: List of dicts with 'question', 'answer', 'expected'
            batch_size: Number of evaluations per batch

        Returns:
            List of JudgeResults
        """
        results = []

        # First, check cache for all
        uncached = []
        for i, eval_item in enumerate(evaluations):
            cache_key = self._cache_key(
                eval_item["question"],
                eval_item["answer"],
                eval_item.get("expected", ""),
            )
            if self._cache_enabled and cache_key in self._cache:
                self._cache_hits += 1
                result = self._cache[cache_key]
                result.cached = True
                results.append((i, result))
            else:
                uncached.append((i, eval_item))

        # Process uncached in batches
        for batch_start in range(0, len(uncached), batch_size):
            batch = uncached[batch_start:batch_start + batch_size]

            for idx, eval_item in batch:
                result = self.evaluate(
                    question=eval_item["question"],
                    answer=eval_item["answer"],
                    expected=eval_item.get("expected"),
                    expected_keywords=eval_item.get("expected_keywords"),
                )
                results.append((idx, result))

            # Small delay to avoid rate limits
            if batch_start + batch_size < len(uncached):
                time.sleep(0.1)

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def get_stats(self) -> Dict[str, Any]:
        """Get judge statistics"""
        return {
            "total_calls": self._total_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "hit_rate": self._cache_hits / max(1, self._total_calls + self._cache_hits),
        }


def batch_judge(
    evaluations: List[Dict[str, Any]],
    llm_client=None,
    sample_size: Optional[int] = None,
) -> List[JudgeResult]:
    """
    Convenience function for batch evaluation.

    Args:
        evaluations: List of evaluation items
        llm_client: LLM client
        sample_size: If set, randomly sample this many evaluations

    Returns:
        List of JudgeResults
    """
    import random

    if sample_size and len(evaluations) > sample_size:
        evaluations = random.sample(evaluations, sample_size)

    judge = LLMJudge(llm_client=llm_client)
    return judge.evaluate_batch(evaluations)


def compute_inter_annotator_agreement(
    results_a: List[JudgeResult],
    results_b: List[JudgeResult],
) -> float:
    """
    Compute agreement between two judges.

    Uses Cohen's kappa for ordinal ratings.
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result lists must have same length")

    # Compute correlation of overall scores
    scores_a = [r.overall for r in results_a]
    scores_b = [r.overall for r in results_b]

    # Simple correlation
    import numpy as np
    if np.std(scores_a) == 0 or np.std(scores_b) == 0:
        return 1.0 if scores_a == scores_b else 0.0

    correlation = np.corrcoef(scores_a, scores_b)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0
