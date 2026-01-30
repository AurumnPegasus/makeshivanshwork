"""
Statistical Tests for Memory Architecture Research

Provides rigorous statistical analysis:
- Paired t-test for system comparison
- Bootstrap confidence intervals
- Significance matrix across systems
- Effect size (Cohen's d)

METHODOLOGY:
- Paired tests because same test cases across systems
- Bootstrap for non-normal distributions
- Multiple comparison correction (Bonferroni)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None


@dataclass
class TestResult:
    """Result of a statistical test"""
    statistic: float
    p_value: float
    significant: bool
    effect_size: float  # Cohen's d
    confidence_interval: Tuple[float, float]
    method: str


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> TestResult:
    """
    Paired t-test for comparing two systems.

    Args:
        scores_a: Scores from system A (per test case)
        scores_b: Scores from system B (per test case)
        alpha: Significance level

    Returns:
        TestResult with p-value and effect size
    """
    if not HAS_SCIPY:
        # Fallback without scipy
        a = np.array(scores_a)
        b = np.array(scores_b)
        diff = a - b
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        return TestResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            effect_size=float(effect_size),
            confidence_interval=(np.mean(diff) - 1.96 * np.std(diff), np.mean(diff) + 1.96 * np.std(diff)),
            method="paired_t_test (scipy not installed)",
        )

    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    a = np.array(scores_a)
    b = np.array(scores_b)

    # Paired t-test
    statistic, p_value = stats.ttest_rel(a, b)

    # Effect size (Cohen's d for paired samples)
    diff = a - b
    effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    # Confidence interval for mean difference
    mean_diff = np.mean(diff)
    se = stats.sem(diff)
    ci = stats.t.interval(1 - alpha, len(diff) - 1, loc=mean_diff, scale=se)

    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(effect_size),
        confidence_interval=(float(ci[0]), float(ci[1])),
        method="paired_t_test",
    )


def bootstrap_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.

    Args:
        scores: Score values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        (mean, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    scores = np.array(scores)

    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(np.mean(scores)), float(ci_lower), float(ci_upper)


def significance_matrix(
    systems: Dict[str, List[float]],
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> Dict[str, Dict[str, TestResult]]:
    """
    Compute pairwise significance tests between all systems.

    Args:
        systems: Dict of system_name → scores
        alpha: Significance level
        correction: Multiple comparison correction (bonferroni, none)

    Returns:
        Matrix of TestResults
    """
    system_names = list(systems.keys())
    n_comparisons = len(system_names) * (len(system_names) - 1) // 2

    # Adjust alpha for multiple comparisons
    if correction == "bonferroni":
        adjusted_alpha = alpha / max(1, n_comparisons)
    else:
        adjusted_alpha = alpha

    matrix = {}
    for name_a in system_names:
        matrix[name_a] = {}
        for name_b in system_names:
            if name_a == name_b:
                matrix[name_a][name_b] = None
            elif name_b in matrix and name_a in matrix[name_b]:
                # Mirror the existing result
                matrix[name_a][name_b] = matrix[name_b][name_a]
            else:
                result = paired_t_test(
                    systems[name_a],
                    systems[name_b],
                    alpha=adjusted_alpha,
                )
                matrix[name_a][name_b] = result

    return matrix


def format_significance_matrix(
    matrix: Dict[str, Dict[str, TestResult]],
    metric_name: str = "Metric",
) -> str:
    """
    Format significance matrix as markdown table.

    Shows:
    - ** if p < 0.01
    - * if p < 0.05
    - ns if not significant
    """
    systems = list(matrix.keys())

    # Build table
    lines = [
        f"## Significance Matrix: {metric_name}",
        "",
        "| | " + " | ".join(systems) + " |",
        "|" + "|".join(["---"] * (len(systems) + 1)) + "|",
    ]

    for name_a in systems:
        row = [name_a]
        for name_b in systems:
            if name_a == name_b:
                row.append("-")
            else:
                result = matrix[name_a][name_b]
                if result is None:
                    row.append("-")
                elif result.p_value < 0.01:
                    row.append(f"**{result.effect_size:.2f}")
                elif result.p_value < 0.05:
                    row.append(f"*{result.effect_size:.2f}")
                else:
                    row.append("ns")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "** p < 0.01, * p < 0.05, ns = not significant",
        "Values show Cohen's d effect size",
    ])

    return "\n".join(lines)


def compute_effect_size(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size with interpretation.

    Returns:
        (effect_size, interpretation)
    """
    a = np.array(scores_a)
    b = np.array(scores_b)

    # Pooled standard deviation
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) /
        (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0, "undefined"

    d = (np.mean(a) - np.mean(b)) / pooled_std

    # Interpretation (Cohen's conventions)
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(d), interpretation


def generate_statistical_report(
    systems: Dict[str, List[float]],
    metric_name: str = "Recall@5",
) -> str:
    """
    Generate complete statistical analysis report.

    Args:
        systems: Dict of system_name → per-case scores
        metric_name: Name of the metric

    Returns:
        Formatted report
    """
    lines = [
        f"# Statistical Analysis: {metric_name}",
        "",
        "## Summary Statistics",
        "",
        "| System | Mean | 95% CI | Std |",
        "|--------|------|--------|-----|",
    ]

    for name, scores in systems.items():
        mean, ci_lo, ci_hi = bootstrap_confidence_interval(scores)
        std = np.std(scores)
        lines.append(f"| {name} | {mean:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {std:.3f} |")

    # Significance matrix
    matrix = significance_matrix(systems)
    lines.append("")
    lines.append(format_significance_matrix(matrix, metric_name))

    # Pairwise comparisons against best
    best_system = max(systems.items(), key=lambda x: np.mean(x[1]))[0]
    lines.extend([
        "",
        f"## Comparisons vs Best ({best_system})",
        "",
    ])

    for name, scores in systems.items():
        if name == best_system:
            continue
        d, interp = compute_effect_size(systems[best_system], scores)
        lines.append(f"- **{name}**: d = {d:.2f} ({interp})")

    return "\n".join(lines)
