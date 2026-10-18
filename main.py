"""RPForest CLI entrypoint and benchmark demonstration."""

from __future__ import annotations

import sys
import time
import numpy as np
from rpforest.exact import BruteForceIndex
from rpforest.forest import RPForest


def run_demo() -> None:
    """Execute a concise demonstration comparing RPForest against BruteForceIndex."""
    print("=== RPForest: Vector Search & Benchmark Demo ===\n")
    np.random.seed(42)
    n_samples, n_dim = 2_000, 32
    n_queries = 50
    k = 10

    print(f"Building dataset: {n_samples} vectors with {n_dim} dimensions...")
    dataset = np.random.randn(n_samples, n_dim).astype(np.float64)
    queries = np.random.randn(n_queries, n_dim).astype(np.float64)

    # 1. Exact BruteForceIndex
    print("\n1. Exact Brute-Force Flat Index:")
    t_start = time.perf_counter()
    exact_index = BruteForceIndex(metric="euclidean").fit(dataset)
    fit_time_bf = (time.perf_counter() - t_start) * 1000.0
    print(f"   Indexed {n_samples} points in {fit_time_bf:.2f} ms")

    t_start = time.perf_counter()
    exact_indices, exact_dists = exact_index.batch_query(queries, k=k)
    exact_query_time = (time.perf_counter() - t_start) * 1000.0
    avg_exact_latency = exact_query_time / n_queries
    print(f"   Queried {n_queries} vectors: {avg_exact_latency:.3f} ms / query (baseline recall = 100%)")

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
    forest_indices, forest_dists = forest.batch_query(queries, k=k)
    forest_query_time = (time.perf_counter() - t_start) * 1000.0
    avg_forest_latency = forest_query_time / n_queries

    # Calculate empirical recall@k
    recalls: list[float] = []
    for i in range(n_queries):
        true_set = set(exact_indices[i].tolist())
        pred_set = set(forest_indices[i].tolist())
        recalls.append(len(true_set & pred_set) / k)
    avg_recall = float(np.mean(recalls))

    print(f"   Queried {n_queries} vectors: {avg_forest_latency:.3f} ms / query")
    print(f"   Empirical Recall@{k}: {avg_recall * 100:.2f}%")

    # Sample query printout
    print("\n3. Sample Query Neighbor Retrieval:")
    print("   Query vector [0] top-5 results:")
    print("   Exact top-5:  ", exact_indices[0][:5], [round(d, 4) for d in exact_dists[0][:5]])
    print("   Forest top-5: ", forest_indices[0][:5], [round(d, 4) for d in forest_dists[0][:5]])

    print("\nDemo completed successfully.")


def main() -> None:
    """CLI entrypoint."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python main.py [demo]")
        print("\nRun RPForest approximate nearest neighbor demonstration and baseline benchmark.")
        sys.exit(0)

    run_demo()


if __name__ == "__main__":
    main()
