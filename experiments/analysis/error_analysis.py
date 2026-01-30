"""
Error Analysis for Memory Architecture Research

Categorizes and analyzes failure modes:
1. Retrieval failures (relevant not retrieved)
2. Ranking failures (relevant retrieved but low rank)
3. Precision failures (irrelevant in top results)
4. Action safety failures (questions triggering actions)
5. Temporal failures (wrong time period)
6. Entity failures (wrong entities)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict


@dataclass
class ErrorCase:
    """A single error case"""
    test_case_id: str
    query: str
    error_type: str
    expected: Any
    actual: Any
    severity: str  # critical, major, minor
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorAnalysis:
    """Complete error analysis results"""
    total_cases: int
    total_errors: int
    error_rate: float
    errors_by_type: Dict[str, int]
    errors_by_severity: Dict[str, int]
    error_cases: List[ErrorCase]

    def summary(self) -> str:
        """Generate summary text"""
        lines = [
            f"Total cases: {self.total_cases}",
            f"Total errors: {self.total_errors} ({self.error_rate:.1%})",
            "",
            "Errors by type:",
        ]
        for error_type, count in sorted(
            self.errors_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {error_type}: {count}")

        lines.extend([
            "",
            "Errors by severity:",
        ])
        for severity, count in self.errors_by_severity.items():
            lines.append(f"  {severity}: {count}")

        return "\n".join(lines)


def analyze_errors(
    test_cases: List[Dict],
    results: List[Dict],
    system_name: str,
) -> ErrorAnalysis:
    """
    Analyze errors from evaluation results.

    Args:
        test_cases: Original test cases
        results: Search results for each test case
        system_name: Name of system being analyzed

    Returns:
        ErrorAnalysis with categorized failures
    """
    errors_by_type = defaultdict(int)
    errors_by_severity = defaultdict(int)
    error_cases = []

    for case, result in zip(test_cases, results):
        # Check for different types of errors
        error = _detect_error(case, result)

        if error:
            errors_by_type[error.error_type] += 1
            errors_by_severity[error.severity] += 1
            error_cases.append(error)

    total_errors = len(error_cases)
    error_rate = total_errors / max(1, len(test_cases))

    return ErrorAnalysis(
        total_cases=len(test_cases),
        total_errors=total_errors,
        error_rate=error_rate,
        errors_by_type=dict(errors_by_type),
        errors_by_severity=dict(errors_by_severity),
        error_cases=error_cases,
    )


def _detect_error(
    case: Dict,
    result: Dict,
) -> Optional[ErrorCase]:
    """Detect error in a single test case"""
    query = case.get("query", "")
    case_type = case.get("type", "")
    case_id = case.get("id", "unknown")

    # Check retrieval errors
    relevant_ids = set(case.get("relevant_message_ids", []))
    retrieved_ids = result.get("retrieved_ids", [])

    if relevant_ids and not relevant_ids & set(retrieved_ids):
        return ErrorCase(
            test_case_id=case_id,
            query=query,
            error_type="retrieval_miss",
            expected=list(relevant_ids),
            actual=retrieved_ids[:5],
            severity="major",
            details={"case_type": case_type},
        )

    # Check ranking errors (relevant found but not in top 5)
    if relevant_ids:
        top5_ids = set(retrieved_ids[:5])
        if relevant_ids & set(retrieved_ids) and not (relevant_ids & top5_ids):
            return ErrorCase(
                test_case_id=case_id,
                query=query,
                error_type="ranking_error",
                expected="Relevant in top 5",
                actual=f"Relevant at positions: {[retrieved_ids.index(r) for r in relevant_ids if r in retrieved_ids]}",
                severity="minor",
                details={"case_type": case_type},
            )

    # Check action safety errors
    if case_type == "question_not_command":
        should_not_trigger = case.get("should_not_trigger", [])
        triggered = result.get("triggered_actions", [])

        if any(action in triggered for action in should_not_trigger):
            return ErrorCase(
                test_case_id=case_id,
                query=query,
                error_type="false_action",
                expected="No action triggered",
                actual=triggered,
                severity="critical",
                details={
                    "case_type": case_type,
                    "should_not_trigger": should_not_trigger,
                },
            )

    # Check for missing command execution
    if case_type == "actual_command":
        should_trigger = case.get("should_trigger", [])
        triggered = result.get("triggered_actions", [])

        if should_trigger and not any(action in triggered for action in should_trigger):
            return ErrorCase(
                test_case_id=case_id,
                query=query,
                error_type="missed_command",
                expected=should_trigger,
                actual=triggered,
                severity="major",
                details={"case_type": case_type},
            )

    # Check temporal errors
    if case_type == "temporal_recall":
        temporal_filter = case.get("temporal_filter")
        if temporal_filter and not result.get("matched_temporal", False):
            return ErrorCase(
                test_case_id=case_id,
                query=query,
                error_type="temporal_mismatch",
                expected=temporal_filter,
                actual=result.get("retrieved_timestamps"),
                severity="minor",
                details={"case_type": case_type},
            )

    return None


def categorize_failures(
    error_cases: List[ErrorCase],
) -> Dict[str, List[ErrorCase]]:
    """
    Categorize errors for deeper analysis.

    Returns dict of category → error cases.
    """
    categories = defaultdict(list)

    for error in error_cases:
        categories[error.error_type].append(error)

        # Additional categorization
        if error.error_type == "retrieval_miss":
            # Subcategorize retrieval misses
            if "entity" in error.details.get("case_type", ""):
                categories["retrieval_miss_entity"].append(error)
            elif "temporal" in error.details.get("case_type", ""):
                categories["retrieval_miss_temporal"].append(error)
            elif "relationship" in error.details.get("case_type", ""):
                categories["retrieval_miss_relationship"].append(error)

        if error.severity == "critical":
            categories["critical"].append(error)

    return dict(categories)


def generate_error_report(
    analysis: ErrorAnalysis,
    max_examples: int = 5,
) -> str:
    """
    Generate detailed error report.

    Args:
        analysis: Error analysis results
        max_examples: Max examples per category

    Returns:
        Formatted report
    """
    lines = [
        "# Error Analysis Report",
        "",
        analysis.summary(),
        "",
        "## Detailed Examples",
        "",
    ]

    categories = categorize_failures(analysis.error_cases)

    for category, errors in sorted(categories.items()):
        if category in ["critical", "retrieval_miss_entity", "retrieval_miss_temporal"]:
            continue  # Skip subcategories in main listing

        lines.append(f"### {category} ({len(errors)} cases)")
        lines.append("")

        for error in errors[:max_examples]:
            lines.append(f"**Query:** {error.query}")
            lines.append(f"- Expected: {error.expected}")
            lines.append(f"- Actual: {error.actual}")
            lines.append(f"- Severity: {error.severity}")
            lines.append("")

    # Critical errors get special attention
    if "critical" in categories:
        lines.append("## CRITICAL ERRORS (Require Immediate Attention)")
        lines.append("")
        for error in categories["critical"]:
            lines.append(f"**{error.test_case_id}:** {error.query}")
            lines.append(f"- Type: {error.error_type}")
            lines.append(f"- Expected: {error.expected}")
            lines.append(f"- Actual: {error.actual}")
            lines.append("")

    return "\n".join(lines)
