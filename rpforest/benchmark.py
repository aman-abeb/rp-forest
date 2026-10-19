"""Reproducible CLI benchmark harness with ASCII recall and latency profiling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import sys
import time
from typing import Any, Literal, Sequence
import numpy as np

from rpforest.exact import BruteForceIndex
from rpforest.forest import RPForest
from rpforest.metrics import Metric

DatasetDistribution = Literal["clustered", "gaussian", "uniform"]


def generate_dataset(
    n_samples: int = 10_000,
    n_dim: int = 32,
    n_queries: int = 100,
    n_clusters: int = 50,
    distribution: DatasetDistribution = "clustered",
    metric: Metric = "euclidean",
    cluster_std: float = 0.5,
    random_state: int | np.random.Generator | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic reference vectors and evaluation query vectors.

    Parameters
    ----------
    n_samples : int, default=10_000
        Total number of reference points to generate.
    n_dim : int, default=32
        Dimensionality of each vector.
    n_queries : int, default=100
        Number of evaluation query vectors.
    n_clusters : int, default=50
        Number of Gaussian cluster centers (used when distribution='clustered').
    distribution : DatasetDistribution, default='clustered'
        Distribution type: 'clustered', 'gaussian', or 'uniform'.
    metric : Metric, default='euclidean'
        Target distance metric. If 'cosine', vectors are normalized to unit L2 norm.
    cluster_std : float, default=0.5
        Standard deviation of Gaussian clusters around centers.
    random_state : int | np.random.Generator | None, default=42
        Random state seed for reproducibility.

    Returns
    -------
    dataset : np.ndarray
        Reference vector matrix of shape (n_samples, n_dim).
    queries : np.ndarray
        Query vector matrix of shape (n_queries, n_dim).
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    if distribution == "clustered":
        n_c = max(1, min(n_clusters, n_samples))
        centers = rng.uniform(-5.0, 5.0, size=(n_c, n_dim))
        dataset_clusters = rng.integers(0, n_c, size=n_samples)
        dataset = centers[dataset_clusters] + rng.normal(0.0, cluster_std, size=(n_samples, n_dim))

        query_clusters = rng.integers(0, n_c, size=n_queries)
        queries = centers[query_clusters] + rng.normal(0.0, cluster_std, size=(n_queries, n_dim))
    elif distribution == "gaussian":
        dataset = rng.normal(0.0, 1.0, size=(n_samples, n_dim))
        queries = rng.normal(0.0, 1.0, size=(n_queries, n_dim))
    elif distribution == "uniform":
        dataset = rng.uniform(-1.0, 1.0, size=(n_samples, n_dim))
        queries = rng.uniform(-1.0, 1.0, size=(n_queries, n_dim))
    else:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. Choose 'clustered', 'gaussian', or 'uniform'."
        )

    dataset = np.ascontiguousarray(dataset, dtype=np.float64)
    queries = np.ascontiguousarray(queries, dtype=np.float64)

    if metric == "cosine":
        d_norms = np.linalg.norm(dataset, axis=1, keepdims=True)
        d_norms[d_norms < 1e-12] = 1.0
        dataset = dataset / d_norms

        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_norms[q_norms < 1e-12] = 1.0
        queries = queries / q_norms

    return dataset, queries


def calculate_recall_at_k(
    true_indices: np.ndarray,
    pred_indices: np.ndarray,
    k: int | None = None,
) -> float:
    """Compute empirical Recall@k of approximate retrieval against exact ground truth.

    Parameters
    ----------
    true_indices : np.ndarray
        Ground truth nearest neighbor indices of shape (n_queries, k) or (k,).
    pred_indices : np.ndarray
        Retrieved approximate nearest neighbor indices of shape (n_queries, k) or (k,).
    k : int | None, default=None
        Depth k to evaluate. If None, uses min columns available.

    Returns
    -------
    float
        Mean recall across queries in range [0.0, 1.0].
    """
    true_arr = np.asarray(true_indices)
    pred_arr = np.asarray(pred_indices)

    if true_arr.ndim == 1 and pred_arr.ndim == 1:
        true_arr = true_arr[None, :]
        pred_arr = pred_arr[None, :]

    if true_arr.ndim != 2 or pred_arr.ndim != 2:
        raise ValueError("true_indices and pred_indices must be 1D or 2D arrays.")

    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError(
            f"Query count mismatch: true has {true_arr.shape[0]}, pred has {pred_arr.shape[0]}"
        )

    n_queries = true_arr.shape[0]
    if n_queries == 0:
        return 0.0

    k_eval = (
        min(true_arr.shape[1], pred_arr.shape[1])
        if k is None
        else min(k, true_arr.shape[1], pred_arr.shape[1])
    )
    if k_eval <= 0:
        return 0.0

    recalls = np.empty(n_queries, dtype=np.float64)
    for i in range(n_queries):
        t_set = set(true_arr[i, :k_eval].tolist())
        p_set = set(pred_arr[i, :k_eval].tolist())
        recalls[i] = len(t_set & p_set) / float(k_eval)

    return float(np.mean(recalls))


@dataclass
class BenchmarkRunResult:
    """Performance metrics for a single index configuration."""

    name: str
    index_type: str
    n_trees: int | None = None
    leaf_size: int | None = None
    search_budget: int | None = None
    build_time_ms: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    qps: float = 0.0
    recall_at_k: float = 1.0
    speedup_vs_baseline: float = 1.0
    avg_candidates_evaluated: float = 0.0


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from a benchmark run."""

    n_samples: int
    n_dim: int
    n_queries: int
    k: int
    metric: str
    baseline: BenchmarkRunResult
    runs: list[BenchmarkRunResult] = field(default_factory=list)

    @property
    def all_runs(self) -> list[BenchmarkRunResult]:
        """All runs including the exact brute-force baseline."""
        return [self.baseline] + self.runs

    def to_dict(self) -> dict[str, Any]:
        """Convert result suite into a serializable dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_dim": self.n_dim,
            "n_queries": self.n_queries,
            "k": self.k,
            "metric": self.metric,
            "baseline": asdict(self.baseline),
            "runs": [asdict(r) for r in self.runs],
        }


def run_benchmark(
    n_samples: int = 10_000,
    n_dim: int = 32,
    n_queries: int = 100,
    k: int = 10,
    metric: Metric = "euclidean",
    n_trees: int = 10,
    leaf_size: int = 40,
    search_budgets: Sequence[int | None] = (100, 200, 350, 500, None),
    n_clusters: int = 50,
    random_state: int = 42,
    warmup_queries: int = 5,
) -> BenchmarkSuiteResult:
    """Execute a full nearest neighbor search benchmark.

    Evaluates exact BruteForceIndex ground truth and profiles RPForest across
    multiple candidate search budgets to capture recall-latency tradeoffs.

    Parameters
    ----------
    n_samples : int, default=10_000
        Number of reference vectors.
    n_dim : int, default=32
        Vector dimension.
    n_queries : int, default=100
        Number of queries to evaluate.
    k : int, default=10
        Number of nearest neighbors to retrieve per query.
    metric : Metric, default="euclidean"
        Distance metric ("euclidean" or "cosine").
    n_trees : int, default=10
        Number of trees in RPForest.
    leaf_size : int, default=40
        Leaf size for RPForest trees.
    search_budgets : Sequence[int | None], default=(100, 200, 350, 500, None)
        Search candidate budget configurations to evaluate.
    n_clusters : int, default=50
        Number of clusters for synthetic clustered distribution.
    random_state : int, default=42
        Seed for reproducible dataset generation and forest construction.
    warmup_queries : int, default=5
        Number of un-timed warmup queries before profiling.

    Returns
    -------
    BenchmarkSuiteResult
        Complete suite of benchmark results and timing metrics.
    """
    dataset, queries = generate_dataset(
        n_samples=n_samples,
        n_dim=n_dim,
        n_queries=n_queries,
        n_clusters=n_clusters,
        distribution="clustered",
        metric=metric,
        random_state=random_state,
    )

    # 1. Profile Exact Brute-Force Baseline
    t_start = time.perf_counter()
    exact_index = BruteForceIndex(metric=metric).fit(dataset)
    bf_build_ms = (time.perf_counter() - t_start) * 1000.0

    # Warmup
    for i in range(min(warmup_queries, n_queries)):
        exact_index.query(queries[i], k=k)

    bf_latencies: list[float] = []
    ground_truth_indices = np.empty((n_queries, k), dtype=np.int64)

    for i in range(n_queries):
        t0 = time.perf_counter()
        idx, _ = exact_index.query(queries[i], k=k)
        t1 = time.perf_counter()
        bf_latencies.append((t1 - t0) * 1000.0)
        ground_truth_indices[i] = idx

    bf_mean_ms = float(np.mean(bf_latencies))
    baseline_result = BenchmarkRunResult(
        name="BruteForceIndex (Exact)",
        index_type="brute_force",
        build_time_ms=bf_build_ms,
        mean_latency_ms=bf_mean_ms,
        p50_latency_ms=float(np.percentile(bf_latencies, 50)),
        p95_latency_ms=float(np.percentile(bf_latencies, 95)),
        p99_latency_ms=float(np.percentile(bf_latencies, 99)),
        qps=1000.0 / bf_mean_ms if bf_mean_ms > 0 else 0.0,
        recall_at_k=1.0,
        speedup_vs_baseline=1.0,
        avg_candidates_evaluated=float(n_samples),
    )

    # 2. Build RPForest Index
    t_start = time.perf_counter()
    forest = RPForest(
        n_trees=n_trees,
        leaf_size=leaf_size,
        metric=metric,
        random_state=random_state,
    ).fit(dataset)
    forest_build_ms = (time.perf_counter() - t_start) * 1000.0

    # 3. Profile RPForest across candidate search budgets
    runs: list[BenchmarkRunResult] = []
    for budget in search_budgets:
        budget_label = f"budget={budget}" if budget is not None else "unbounded"
        name = f"RPForest ({budget_label})"

        # Warmup
        for i in range(min(warmup_queries, n_queries)):
            forest.query(queries[i], k=k, search_budget=budget)

        latencies: list[float] = []
        pred_indices = np.empty((n_queries, k), dtype=np.int64)
        cand_counts: list[int] = []

        for i in range(n_queries):
            q_vec = queries[i]
            # Time individual end-to-end query
            t0 = time.perf_counter()
            idx, _ = forest.query(q_vec, k=k, search_budget=budget)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            pred_indices[i] = idx

            # Track candidate pool size
            cands = forest.get_candidates(q_vec, search_budget=budget)
            cand_counts.append(len(cands))

        mean_ms = float(np.mean(latencies))
        recall = calculate_recall_at_k(ground_truth_indices, pred_indices, k=k)
        speedup = bf_mean_ms / mean_ms if mean_ms > 0 else 1.0

        runs.append(
            BenchmarkRunResult(
                name=name,
                index_type="rpforest",
                n_trees=n_trees,
                leaf_size=leaf_size,
                search_budget=budget,
                build_time_ms=forest_build_ms,
                mean_latency_ms=mean_ms,
                p50_latency_ms=float(np.percentile(latencies, 50)),
                p95_latency_ms=float(np.percentile(latencies, 95)),
                p99_latency_ms=float(np.percentile(latencies, 99)),
                qps=1000.0 / mean_ms if mean_ms > 0 else 0.0,
                recall_at_k=recall,
                speedup_vs_baseline=speedup,
                avg_candidates_evaluated=float(np.mean(cand_counts)),
            )
        )

    return BenchmarkSuiteResult(
        n_samples=n_samples,
        n_dim=n_dim,
        n_queries=n_queries,
        k=k,
        metric=metric,
        baseline=baseline_result,
        runs=runs,
    )


def check_success_criteria(
    result: BenchmarkSuiteResult,
    target_recall: float = 0.90,
    target_speedup: float = 5.0,
) -> tuple[bool, str]:
    """Check if any RPForest run meets or exceeds the target recall and speedup thresholds.

    Parameters
    ----------
    result : BenchmarkSuiteResult
        Completed benchmark suite results.
    target_recall : float, default=0.90
        Minimum required recall@k (e.g. 90%).
    target_speedup : float, default=5.0
        Minimum required speedup multiplier over brute force (e.g. 5x).

    Returns
    -------
    passed : bool
        True if at least one configuration satisfies both criteria.
    message : str
        Detailed status message describing the outcome.
    """
    qualifying = [
        run
        for run in result.runs
        if run.recall_at_k >= target_recall and run.speedup_vs_baseline >= target_speedup
    ]

    if qualifying:
        # Select the run with the highest speedup among qualifying runs
        best = max(qualifying, key=lambda r: r.speedup_vs_baseline)
        msg = (
            f"SUCCESS: {best.name} achieved {best.recall_at_k * 100:.2f}% recall@{result.k} "
            f"at {best.speedup_vs_baseline:.2f}x speedup vs baseline "
            f"(target: >={target_recall * 100:.1f}% recall and >={target_speedup:.1f}x speedup)."
        )
        return True, msg

    # If none met both, find max recall and max speedup to provide actionable diagnostics
    max_rec = max((r.recall_at_k for r in result.runs), default=0.0)
    max_spd = max((r.speedup_vs_baseline for r in result.runs), default=0.0)
    msg = (
        f"FAIL: No configuration met both criteria simultaneously. "
        f"Peak recall: {max_rec * 100:.2f}% (target >={target_recall * 100:.1f}%), "
        f"Peak speedup: {max_spd:.2f}x (target >={target_speedup:.1f}x)."
    )
    return False, msg


def generate_ascii_table(result: BenchmarkSuiteResult) -> str:
    """Format benchmark metrics into an ASCII comparison table."""
    headers = [
        "Method / Configuration",
        f"Recall@{result.k}",
        "Mean (ms)",
        "P95 (ms)",
        "P99 (ms)",
        "QPS",
        "Speedup",
    ]

    rows: list[list[str]] = []
    for run in result.all_runs:
        recall_str = f"{run.recall_at_k * 100:.1f}%"
        mean_str = f"{run.mean_latency_ms:.3f}"
        p95_str = f"{run.p95_latency_ms:.3f}"
        p99_str = f"{run.p99_latency_ms:.3f}"
        qps_str = f"{run.qps:.1f}"
        speedup_str = f"{run.speedup_vs_baseline:.2f}x"
        rows.append([
            run.name,
            recall_str,
            mean_str,
            p95_str,
            p99_str,
            qps_str,
            speedup_str,
        ])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines: list[str] = [sep]
    header_row = (
        "| "
        + " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
        + " |"
    )
    lines.append(header_row)
    lines.append(sep)

    for row in rows:
        row_str = "| "
        for i, cell in enumerate(row):
            if i == 0:
                row_str += f"{cell:<{col_widths[i]}} | "
            else:
                row_str += f"{cell:>{col_widths[i]}} | "
        lines.append(row_str.rstrip())

    lines.append(sep)
    return "\n".join(lines)


def generate_ascii_tradeoff_plot(
    result: BenchmarkSuiteResult,
    width: int = 56,
    height: int = 13,
) -> str:
    """Generate an ASCII scatter plot visualizing Recall@k vs. Mean Query Latency."""
    runs = result.runs
    if not runs:
        return "No benchmark runs to plot."

    # Determine bounds
    all_latencies = [r.mean_latency_ms for r in result.all_runs]
    all_recalls = [r.recall_at_k for r in runs]

    min_x = min(all_latencies)
    max_x = max(all_latencies)
    if max_x == min_x:
        max_x = min_x + 0.1

    # Y bounds from min recall to 1.0
    min_y = max(0.0, min(all_recalls) - 0.05)
    max_y = 1.02
    if max_y <= min_y:
        max_y = min_y + 0.1

    plot_w = max(20, width)
    plot_h = max(8, height)

    grid = [[" " for _ in range(plot_w)] for _ in range(plot_h)]

    # Draw horizontal target line at 0.90 recall
    target_y = 0.90
    if min_y <= target_y <= max_y:
        row_idx = int(round((max_y - target_y) / (max_y - min_y) * (plot_h - 1)))
        row_idx = max(0, min(plot_h - 1, row_idx))
        for col in range(plot_w):
            grid[row_idx][col] = "-"

    # Plot points
    markers: list[tuple[str, BenchmarkRunResult]] = []
    for idx, run in enumerate(runs, start=1):
        marker = str(idx)
        markers.append((marker, run))

        # Map coords
        x_norm = (run.mean_latency_ms - min_x) / (max_x - min_x)
        y_norm = (max_y - run.recall_at_k) / (max_y - min_y)

        col = int(round(x_norm * (plot_w - 1)))
        row = int(round(y_norm * (plot_h - 1)))
        col = max(0, min(plot_w - 1, col))
        row = max(0, min(plot_h - 1, row))
        grid[row][col] = marker

    # Baseline marker
    bf = result.baseline
    bf_x_norm = (bf.mean_latency_ms - min_x) / (max_x - min_x)
    bf_y_norm = (max_y - bf.recall_at_k) / (max_y - min_y)
    bf_col = max(0, min(plot_w - 1, int(round(bf_x_norm * (plot_w - 1)))))
    bf_row = max(0, min(plot_h - 1, int(round(bf_y_norm * (plot_h - 1)))))
    grid[bf_row][bf_col] = "B"

    # Render with Y-axis labels
    lines: list[str] = []
    lines.append(" " * 8 + f"Recall@{result.k} vs. Mean Query Latency (ms)")
    lines.append(" " * 8 + "+" + "-" * plot_w + "+")

    for r_idx in range(plot_h):
        y_val = max_y - (r_idx / (plot_h - 1)) * (max_y - min_y)
        label = f"{y_val * 100:5.1f}% |"
        row_str = "".join(grid[r_idx])
        suffix = ""
        if r_idx == row_idx and min_y <= target_y <= max_y:
            suffix = "  <- 90% Recall Target"
        lines.append(f"{label}{row_str}|{suffix}")

    lines.append(" " * 8 + "+" + "-" * plot_w + "+")
    x_axis_label = f"        {min_x:.3f} ms" + " " * max(1, plot_w - 20) + f"{max_x:.3f} ms"
    lines.append(x_axis_label)
    lines.append(" " * (plot_w // 2) + "Mean Latency (Lower is faster)")

    # Legend
    lines.append("\nPlot Legend:")
    lines.append(f"  [B] {bf.name} (Latency: {bf.mean_latency_ms:.3f} ms, Recall: 100.0%)")
    for marker, run in markers:
        lines.append(
            f"  [{marker}] {run.name}: Recall={run.recall_at_k * 100:.1f}%, "
            f"Latency={run.mean_latency_ms:.3f} ms, Speedup={run.speedup_vs_baseline:.2f}x"
        )

    return "\n".join(lines)


def format_benchmark_report(result: BenchmarkSuiteResult, include_plot: bool = True) -> str:
    """Compose the full text benchmark report containing summary, table, and ASCII plot."""
    passed, criteria_msg = check_success_criteria(result)

    divider = "=" * 78
    subdivider = "-" * 78
    lines = [
        divider,
        "               RPForest Vector Search Benchmark Report",
        divider,
        f"Dataset:      {result.n_samples:,} vectors, {result.n_dim} dimensions",
        f"Queries:      {result.n_queries} queries, top-{result.k} retrieval (metric: {result.metric})",
        f"Index Config: {result.runs[0].n_trees if result.runs else 'N/A'} trees, "
        f"leaf_size={result.runs[0].leaf_size if result.runs else 'N/A'}",
        subdivider,
        "1. Recall & Latency Comparison Table:",
        generate_ascii_table(result),
        "",
    ]

    if include_plot:
        lines.extend([
            "2. ASCII Recall vs. Latency Tradeoff Curve:",
            generate_ascii_tradeoff_plot(result),
            "",
        ])

    lines.extend([
        "3. Target Verification Check:",
        f"   Status: {'[PASS]' if passed else '[FAIL]'} {criteria_msg}",
        divider,
    ])

    return "\n".join(lines)


def parse_budgets(val: str) -> list[int | None]:
    """Parse comma-separated candidate budgets from CLI string."""
    items = [item.strip() for item in val.split(",") if item.strip()]
    budgets: list[int | None] = []
    for item in items:
        if item.lower() in ("none", "null", "unbounded", "all"):
            budgets.append(None)
        else:
            budgets.append(int(item))
    return budgets


def build_benchmark_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for benchmark harness."""
    parser = argparse.ArgumentParser(
        prog="rpforest-benchmark",
        description="Benchmark RPForest against BruteForceIndex with ASCII reports.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of reference vectors in synthetic dataset (default: 10000).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Dimensionality of vectors (default: 32).",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of query vectors (default: 100).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k neighbors to retrieve (default: 10).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric (default: euclidean).",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=10,
        help="Number of random projection trees (default: 10).",
    )
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=40,
        help="Maximum leaf size before splitting (default: 40).",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="100,200,350,500,none",
        help="Comma-separated candidate search budgets (default: 100,200,350,500,none).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=50,
        help="Number of clusters for clustered synthetic distribution (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the ASCII tradeoff curve in the report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw benchmark results as a JSON object.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if success criteria (>90%% recall and >=5x speedup) are not met.",
    )
    return parser


def benchmark_cli(args: Sequence[str] | None = None) -> int:
    """CLI runner for the benchmark harness."""
    parser = build_benchmark_parser()
    parsed = parser.parse_args(args)

    budgets = parse_budgets(parsed.budgets)

    result = run_benchmark(
        n_samples=parsed.n_samples,
        n_dim=parsed.dim,
        n_queries=parsed.n_queries,
        k=parsed.k,
        metric=parsed.metric,
        n_trees=parsed.trees,
        leaf_size=parsed.leaf_size,
        search_budgets=budgets,
        n_clusters=parsed.clusters,
        random_state=parsed.seed,
    )

    if parsed.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        report = format_benchmark_report(result, include_plot=not parsed.no_plot)
        print(report)

    if parsed.check:
        passed, _ = check_success_criteria(result)
        return 0 if passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(benchmark_cli())
