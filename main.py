"""RPForest CLI entrypoint and benchmark demonstration."""

from __future__ import annotations

import sys
import time
import numpy as np

from rpforest.benchmark import benchmark_cli
from rpforest.exact import BruteForceIndex
from rpforest.forest import RPForest


def run_demo() -> None:
    """Execute a concise demonstration comparing RPForest against BruteForceIndex."""
    print("=== RPForest: Approximate Nearest Neighbors Demo ===\n")
    np.random.seed(42)
    n_samples, n_dim = 2_000, 32
    n_queries = 30
    k = 10

    print(f"Building dataset: {n_samples:,} vectors with {n_dim} dimensions...")
    dataset = np.random.randn(n_samples, n_dim).astype(np.float64)
    queries = np.random.randn(n_queries, n_dim).astype(np.float64)

    # 1. Exact BruteForceIndex
    print("\n1. Exact Brute-Force Flat Index Baseline:")
    t_start = time.perf_counter()
    exact_index = BruteForceIndex(metric="euclidean").fit(dataset)
    fit_time_bf = (time.perf_counter() - t_start) * 1000.0
    print(f"   Indexed {n_samples:,} vectors in {fit_time_bf:.2f} ms")

    t_start = time.perf_counter()
    exact_indices, exact_dists = exact_index.batch_query(queries, k=k)
    exact_query_time = (time.perf_counter() - t_start) * 1000.0
    avg_exact_latency = exact_query_time / n_queries
    print(f"   Batch queried {n_queries} vectors: {avg_exact_latency:.3f} ms / query (baseline recall = 100%)")

    # 2. Approximate RPForest Index
    print("\n2. Random Projection Forest Index:")
    n_trees = 10
    leaf_size = 25
    t_start = time.perf_counter()
    forest = RPForest(
        n_trees=n_trees,
        leaf_size=leaf_size,
        metric="euclidean",
        random_state=42,
    ).fit(dataset)
    fit_time_forest = (time.perf_counter() - t_start) * 1000.0
    print(f"   Built forest ({n_trees} trees, leaf_size={leaf_size}) in {fit_time_forest:.2f} ms")

    t_start = time.perf_counter()
    forest_indices, forest_dists = forest.batch_query(queries, k=k, search_budget=150)
    forest_query_time = (time.perf_counter() - t_start) * 1000.0
    avg_forest_latency = forest_query_time / n_queries

    # Empirical recall@k
    recalls: list[float] = []
    for i in range(n_queries):
        true_set = set(exact_indices[i].tolist())
        pred_set = set(forest_indices[i].tolist())
        recalls.append(len(true_set & pred_set) / k)
    avg_recall = float(np.mean(recalls))
    speedup = avg_exact_latency / avg_forest_latency if avg_forest_latency > 0 else 1.0

    print(f"   Queried {n_queries} vectors (search_budget=150): {avg_forest_latency:.3f} ms / query")
    print(f"   Empirical Recall@{k}: {avg_recall * 100:.2f}%")
    print(f"   Query Speedup:       {speedup:.2f}x faster than exhaustive flat search")

    # Sample query printout
    print("\n3. Sample Query Neighbor Retrieval:")
    print("   Query vector [0] top-5 results:")
    print("   Exact top-5:  ", exact_indices[0][:5], [round(d, 4) for d in exact_dists[0][:5]])
    print("   Forest top-5: ", forest_indices[0][:5], [round(d, 4) for d in forest_dists[0][:5]])

    print("\n" + "=" * 70)
    print("Full 10,000-vector benchmark harness available via CLI:")
    print("  python main.py benchmark --n-samples 10000 --dim 32 --k 10")
    print("=" * 70)


def main() -> None:
    """CLI entrypoint dispatching between demo and benchmark commands."""
    args = sys.argv[1:]

    if not args:
        # No arguments: execute quick demo
        run_demo()
        return

    if args[0] in ("-h", "--help"):
        print("RPForest: Approximate Nearest Neighbor Search with Random Projection Trees\n")
        print("Usage:")
        print("  python main.py                     Run fast interactive demo")
        print("  python main.py demo                Run fast interactive demo")
        print("  python main.py benchmark [options] Run reproducible 10k-vector CLI benchmark\n")
        print("Benchmark options include:")
        print("  --n-samples INT   Number of vectors (default: 10000)")
        print("  --dim INT         Vector dimension (default: 32)")
        print("  --n-queries INT   Evaluation queries (default: 100)")
        print("  --k INT           Top-k neighbors (default: 10)")
        print("  --trees INT       Number of trees (default: 10)")
        print("  --budgets STR     Search budgets (e.g. 100,200,350,500,none)")
        print("  --json            Output raw results as JSON")
        print("  --check           Assert success criteria (>90% recall@10 and >=5x speedup)")
        return

    if args[0] == "demo":
        run_demo()
        return

    if args[0] == "benchmark":
        sys.exit(benchmark_cli(args[1:]))

    # Default to passing all arguments directly into the benchmark runner
    sys.exit(benchmark_cli(args))


if __name__ == "__main__":
    main()
