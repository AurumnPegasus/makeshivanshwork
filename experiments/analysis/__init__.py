"""
Analysis Tools for Memory Architecture Research

Provides visualization, statistical analysis, and ablation studies.
"""

from .visualize_results import (
    plot_comparison_chart,
    plot_radar_chart,
    plot_latency_distribution,
    generate_results_table,
)
from .ablation_study import (
    run_ablation_study,
    AblationResult,
)
from .error_analysis import (
    analyze_errors,
    categorize_failures,
    ErrorAnalysis,
)
from .statistical_tests import (
    paired_t_test,
    bootstrap_confidence_interval,
    significance_matrix,
)

__all__ = [
    "plot_comparison_chart",
    "plot_radar_chart",
    "plot_latency_distribution",
    "generate_results_table",
    "run_ablation_study",
    "AblationResult",
    "analyze_errors",
    "categorize_failures",
    "ErrorAnalysis",
    "paired_t_test",
    "bootstrap_confidence_interval",
    "significance_matrix",
]
