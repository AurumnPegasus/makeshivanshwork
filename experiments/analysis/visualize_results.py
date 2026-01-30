"""
Visualization Tools for Memory Architecture Research

Generates publication-quality figures:
- Bar charts comparing systems
- Radar charts for multi-metric comparison
- Latency distribution plots
- Results tables (LaTeX/Markdown)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def plot_comparison_chart(
    results: List[Dict[str, Any]],
    metric: str = "recall@5",
    output_path: Optional[Path] = None,
    title: str = "System Comparison",
    figsize: tuple = (10, 6),
):
    """
    Create bar chart comparing systems on a single metric.

    Args:
        results: List of evaluation results
        metric: Metric to compare
        output_path: Where to save figure
        title: Chart title
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    # Extract data
    systems = [r["system_name"] for r in results]

    # Get metric value (handle nested dicts)
    values = []
    for r in results:
        if "." in metric:
            parts = metric.split(".")
            val = r
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else 0
            values.append(float(val) if val else 0)
        elif metric in r.get("retrieval", {}):
            values.append(r["retrieval"][metric])
        elif metric in r.get("answer_quality", {}):
            values.append(r["answer_quality"][metric])
        else:
            values.append(r.get(metric, 0))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color by performance
    colors = plt.cm.RdYlGn(np.array(values) / max(values) if max(values) > 0 else np.zeros_like(values))

    bars = ax.bar(systems, values, color=colors)

    # Styling
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.1 if values else 1)

    # Rotate labels if needed
    if len(systems) > 5:
        plt.xticks(rotation=45, ha="right")

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_radar_chart(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Multi-Metric Comparison",
    figsize: tuple = (10, 8),
):
    """
    Create radar/spider chart for multi-metric comparison.

    Args:
        results: List of evaluation results
        metrics: Metrics to include (default: key metrics)
        output_path: Where to save figure
        title: Chart title
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return

    # Default metrics
    if metrics is None:
        metrics = [
            "retrieval.recall@5",
            "retrieval.mrr",
            "retrieval.precision@5",
            "answer_quality.overall",
            "latency_p50_ms",
        ]

    # Normalize metric names for display
    labels = [m.split(".")[-1] for m in metrics]

    # Extract values for each system
    systems = []
    all_values = []

    for r in results:
        systems.append(r["system_name"])
        values = []
        for metric in metrics:
            if "." in metric:
                parts = metric.split(".")
                val = r
                for p in parts:
                    val = val.get(p, {}) if isinstance(val, dict) else 0
                values.append(float(val) if val else 0)
            else:
                values.append(r.get(metric, 0))
        all_values.append(values)

    # Normalize values (0-1)
    all_values = np.array(all_values)
    for i in range(len(metrics)):
        col = all_values[:, i]
        if "latency" in metrics[i].lower():
            # Lower is better for latency
            if col.max() > 0:
                all_values[:, i] = 1 - (col / col.max())
        else:
            # Higher is better
            if col.max() > 0:
                all_values[:, i] = col / col.max()

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))

    for i, (system, values) in enumerate(zip(systems, all_values)):
        values = values.tolist() + values[:1].tolist()
        ax.plot(angles, values, "o-", linewidth=2, label=system, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_latency_distribution(
    latencies: Dict[str, List[float]],
    output_path: Optional[Path] = None,
    title: str = "Latency Distribution",
    figsize: tuple = (10, 6),
):
    """
    Create latency distribution plot (box plot or violin).

    Args:
        latencies: Dict of system_name → list of latencies
        output_path: Where to save figure
        title: Chart title
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=figsize)

    systems = list(latencies.keys())
    data = [latencies[s] for s in systems]

    # Create box plot
    bp = ax.boxplot(data, labels=systems, patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(systems)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)

    # Add median labels
    for i, d in enumerate(data):
        median = np.median(d)
        ax.annotate(
            f"{median:.0f}ms",
            xy=(i + 1, median),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
        )

    if len(systems) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_results_table(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    format: str = "markdown",  # markdown, latex, csv
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate results table in various formats.

    Args:
        results: List of evaluation results
        metrics: Metrics to include
        format: Output format
        output_path: Where to save table

    Returns:
        Formatted table string
    """
    if metrics is None:
        metrics = [
            ("retrieval.recall@5", "R@5"),
            ("retrieval.mrr", "MRR"),
            ("answer_quality.overall", "Ans"),
            ("false_action_rate", "FAR"),
            ("latency_p50_ms", "P50"),
        ]

    # Build table data
    headers = ["System"] + [m[1] if isinstance(m, tuple) else m for m in metrics]
    rows = []

    for r in results:
        row = [r["system_name"]]
        for metric in metrics:
            metric_key = metric[0] if isinstance(metric, tuple) else metric

            if "." in metric_key:
                parts = metric_key.split(".")
                val = r
                for p in parts:
                    val = val.get(p, {}) if isinstance(val, dict) else 0
                row.append(f"{float(val):.3f}" if val else "N/A")
            else:
                val = r.get(metric_key, "N/A")
                row.append(f"{val:.3f}" if isinstance(val, float) else str(val))

        rows.append(row)

    # Format output
    if format == "markdown":
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        output = "\n".join(lines)

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
        output = "\n".join(lines)

    elif format == "csv":
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(row))
        output = "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {format}")

    if output_path:
        output_path.write_text(output)
        print(f"Saved to {output_path}")

    return output


def create_all_visualizations(
    results_path: Path,
    output_dir: Path,
):
    """
    Create all visualizations from results file.

    Args:
        results_path: Path to results JSON
        output_dir: Directory for outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    # Bar charts for key metrics
    for metric in ["retrieval.recall@5", "retrieval.mrr", "false_action_rate"]:
        name = metric.replace(".", "_").replace("@", "_at_")
        plot_comparison_chart(
            results,
            metric=metric,
            output_path=output_dir / f"bar_{name}.png",
            title=f"Comparison: {metric}",
        )

    # Radar chart
    plot_radar_chart(
        results,
        output_path=output_dir / "radar_comparison.png",
    )

    # Results table
    for fmt in ["markdown", "latex"]:
        table = generate_results_table(results, format=fmt)
        output_path = output_dir / f"results_table.{fmt}"
        output_path.write_text(table)

    print(f"All visualizations saved to {output_dir}")
