"""Unit tests for benchmark harness, metric calculations, and ASCII reports."""

import json
import numpy as np
import pytest

from rpforest.benchmark import (
    BenchmarkRunResult,
    BenchmarkSuiteResult,
    calculate_recall_at_k,
    check_success_criteria,
    format_benchmark_report,
    generate_ascii_table,
    generate_ascii_tradeoff_plot,
    generate_dataset,
    parse_budgets,
    run_benchmark,
)


def test_calculate_recall_at_k_exact_and_partial():
    """Verify recall calculation on synthetic ground truth arrays."""
    # 1. Exact match
    true_idx = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    pred_idx = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    assert np.isclose(calculate_recall_at_k(true_idx, pred_idx, k=4), 1.0)

    # 2. 50% overlap
    pred_half = np.array([[0, 1, 9, 8], [4, 5, 9, 8]])
    assert np.isclose(calculate_recall_at_k(true_idx, pred_half, k=4), 0.5)

    # 3. Disjoint
    pred_none = np.array([[8, 9, 10, 11], [8, 9, 10, 11]])
    assert np.isclose(calculate_recall_at_k(true_idx, pred_none, k=4), 0.0)

    # 4. Truncation to k=2
    pred_first2 = np.array([[0, 1, 99, 99], [4, 5, 99, 99]])
    assert np.isclose(calculate_recall_at_k(true_idx, pred_first2, k=2), 1.0)


def test_calculate_recall_edge_cases():
    """Verify edge-case handling in recall calculation."""
    # 1D vector inputs
    t_1d = np.array([0, 1, 2])
    p_1d = np.array([0, 1, 9])
    assert np.isclose(calculate_recall_at_k(t_1d, p_1d, k=3), 2.0 / 3.0)

    # Zero k or empty queries
    assert calculate_recall_at_k(np.empty((0, 5)), np.empty((0, 5)), k=5) == 0.0
    assert calculate_recall_at_k(np.array([[1, 2]]), np.array([[1, 2]]), k=0) == 0.0

    # Dimension mismatch
    with pytest.raises(ValueError, match="Query count mismatch"):
        calculate_recall_at_k(np.zeros((2, 3)), np.zeros((3, 3)))


def test_generate_dataset_shapes_and_reproducibility():
    """Verify synthetic data generator produces correct shapes and deterministic output."""
    d1, q1 = generate_dataset(n_samples=200, n_dim=16, n_queries=15, random_state=42)
    d2, q2 = generate_dataset(n_samples=200, n_dim=16, n_queries=15, random_state=42)

    assert d1.shape == (200, 16)
    assert q1.shape == (15, 16)
    assert np.array_equal(d1, d2)
    assert np.array_equal(q1, q2)


def test_generate_dataset_distributions_and_cosine():
    """Verify distributions and cosine normalization."""
    for dist in ("clustered", "gaussian", "uniform"):
        d, q = generate_dataset(
            n_samples=50,
            n_dim=8,
            n_queries=5,
            distribution=dist,
            metric="cosine",
            random_state=123,
        )
        assert d.shape == (50, 8)
        assert q.shape == (5, 8)
        # Vectors must be unit normalized for cosine metric
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-6)
        assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)

    with pytest.raises(ValueError, match="Unsupported distribution"):
        generate_dataset(distribution="invalid")  # type: ignore


def test_parse_budgets():
    """Verify CLI budget string parsing."""
    assert parse_budgets("100,200,none") == [100, 200, None]
    assert parse_budgets("50, 150, null, unbounded") == [50, 150, None, None]
    assert parse_budgets("") == []


def test_run_benchmark_small_synthetic():
    """Execute an end-to-end benchmark run on a small dataset."""
    result = run_benchmark(
        n_samples=300,
        n_dim=8,
        n_queries=15,
        k=5,
        metric="euclidean",
        n_trees=6,
        leaf_size=20,
        search_budgets=[25, None],
        n_clusters=10,
        random_state=42,
        warmup_queries=2,
    )

    assert isinstance(result, BenchmarkSuiteResult)
    assert result.n_samples == 300
    assert result.n_dim == 8
    assert result.n_queries == 15
    assert result.k == 5
    assert result.baseline.recall_at_k == 1.0
    assert result.baseline.speedup_vs_baseline == 1.0
    assert result.baseline.qps > 0.0

    assert len(result.runs) == 2
    for run in result.runs:
        assert 0.0 <= run.recall_at_k <= 1.0
        assert run.mean_latency_ms > 0.0
        assert run.qps > 0.0
        assert run.speedup_vs_baseline > 0.0
        assert run.avg_candidates_evaluated > 0.0

    # Test dictionary export
    as_dict = result.to_dict()
    assert as_dict["n_samples"] == 300
    assert len(as_dict["runs"]) == 2
    # Must be valid json
    json_str = json.dumps(as_dict)
    assert len(json_str) > 0


def test_ascii_table_formatting():
    """Verify ASCII table generation contains headers, rows, and formatted data."""
    bf = BenchmarkRunResult(
        name="BruteForceIndex (Exact)",
        index_type="brute_force",
        mean_latency_ms=1.20,
        p95_latency_ms=1.40,
        p99_latency_ms=1.60,
        qps=833.3,
        recall_at_k=1.0,
        speedup_vs_baseline=1.0,
    )
    r1 = BenchmarkRunResult(
        name="RPForest (budget=200)",
        index_type="rpforest",
        mean_latency_ms=0.15,
        p95_latency_ms=0.18,
        p99_latency_ms=0.22,
        qps=6666.7,
        recall_at_k=0.925,
        speedup_vs_baseline=8.0,
    )
    suite = BenchmarkSuiteResult(
        n_samples=5000,
        n_dim=16,
        n_queries=50,
        k=10,
        metric="euclidean",
        baseline=bf,
        runs=[r1],
    )

    table = generate_ascii_table(suite)
    assert "Method / Configuration" in table
    assert "Recall@10" in table
    assert "BruteForceIndex (Exact)" in table
    assert "RPForest (budget=200)" in table
    assert "92.5%" in table
    assert "8.00x" in table


def test_ascii_tradeoff_plot():
    """Verify ASCII tradeoff plot renders axes, labels, and run markers."""
    bf = BenchmarkRunResult(
        name="BruteForceIndex",
        index_type="brute_force",
        mean_latency_ms=1.00,
        recall_at_k=1.0,
    )
    r1 = BenchmarkRunResult(
        name="RPForest (b=100)",
        index_type="rpforest",
        mean_latency_ms=0.10,
        recall_at_k=0.85,
        speedup_vs_baseline=10.0,
    )
    r2 = BenchmarkRunResult(
        name="RPForest (b=300)",
        index_type="rpforest",
        mean_latency_ms=0.18,
        recall_at_k=0.94,
        speedup_vs_baseline=5.5,
    )
    suite = BenchmarkSuiteResult(
        n_samples=5000,
        n_dim=16,
        n_queries=50,
        k=10,
        metric="euclidean",
        baseline=bf,
        runs=[r1, r2],
    )

    plot_str = generate_ascii_tradeoff_plot(suite)
    assert "Recall@10 vs. Mean Query Latency" in plot_str
    assert "[B]" in plot_str
    assert "[1]" in plot_str
    assert "[2]" in plot_str
    assert "90% Recall Target" in plot_str


def test_check_success_criteria_logic():
    """Verify success criteria validation logic for passing and failing runs."""
    bf = BenchmarkRunResult(name="BF", index_type="brute_force", mean_latency_ms=1.0)
    # Passing run: recall >= 90% and speedup >= 5x
    pass_run = BenchmarkRunResult(
        name="RPForest (b=300)",
        index_type="rpforest",
        recall_at_k=0.95,
        speedup_vs_baseline=6.2,
    )
    suite_pass = BenchmarkSuiteResult(
        n_samples=10000,
        n_dim=32,
        n_queries=100,
        k=10,
        metric="euclidean",
        baseline=bf,
        runs=[pass_run],
    )
    passed, msg = check_success_criteria(suite_pass, target_recall=0.90, target_speedup=5.0)
    assert passed is True
    assert "SUCCESS" in msg
    assert "95.00%" in msg

    # Failing run: recall high but speedup too low
    slow_run = BenchmarkRunResult(
        name="RPForest (slow)",
        index_type="rpforest",
        recall_at_k=0.98,
        speedup_vs_baseline=2.5,
    )
    suite_fail = BenchmarkSuiteResult(
        n_samples=10000,
        n_dim=32,
        n_queries=100,
        k=10,
        metric="euclidean",
        baseline=bf,
        runs=[slow_run],
    )
    failed, fail_msg = check_success_criteria(suite_fail, target_recall=0.90, target_speedup=5.0)
    assert failed is False
    assert "FAIL" in fail_msg


def test_format_benchmark_report_content():
    """Verify complete report string contains table, plot, and target summary."""
    bf = BenchmarkRunResult(name="BF", index_type="brute_force", mean_latency_ms=1.0)
    r1 = BenchmarkRunResult(
        name="RPForest (b=200)",
        index_type="rpforest",
        recall_at_k=0.92,
        speedup_vs_baseline=7.1,
    )
    suite = BenchmarkSuiteResult(
        n_samples=2000,
        n_dim=16,
        n_queries=20,
        k=10,
        metric="euclidean",
        baseline=bf,
        runs=[r1],
    )
    report = format_benchmark_report(suite, include_plot=True)
    assert "RPForest Vector Search Benchmark Report" in report
    assert "Recall & Latency Comparison Table" in report
    assert "ASCII Recall vs. Latency Tradeoff Curve" in report
    assert "Target Verification Check" in report
    assert "[PASS]" in report


def test_benchmark_10k_vector_criteria_achievement():
    """Verify RPForest achieves >90% recall@10 with >=5x speedup on 10,000 vectors."""
    result = run_benchmark(
        n_samples=10_000,
        n_dim=32,
        n_queries=30,
        k=10,
        metric="euclidean",
        n_trees=10,
        leaf_size=40,
        search_budgets=[150, 250, 350, None],
        n_clusters=50,
        random_state=42,
        warmup_queries=3,
    )

    passed, msg = check_success_criteria(result, target_recall=0.90, target_speedup=5.0)
    assert passed is True, f"Expected 10,000-vector benchmark to meet criteria: {msg}"
