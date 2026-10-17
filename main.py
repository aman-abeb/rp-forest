"""RPForest CLI entrypoint and quickstart demonstration."""

from __future__ import annotations

import sys
import time
import numpy as np
from rpforest.exact import BruteForceIndex


def run_demo() -> None:
    """Execute a concise demonstration of BruteForceIndex search."""
    print("=== RPForest: Vector Search Demo ===")
    np.random.seed(42)
    n_samples, n_dim = 1_000, 64
    print(f"Generating {n_samples} random vectors with dimension {n_dim}...")
    dataset = np.random.randn(n_samples, n_dim).astype(np.float32)
    query_vec = np.random.randn(n_dim).astype(np.float32)

    # Build and query Euclidean index
    print("\n1. Euclidean Distance Index:")
    euc_index = BruteForceIndex(metric="euclidean").fit(dataset)
    t0 = time.perf_counter()
    indices, dists = euc_index.query(query_vec, k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Retrieved top 5 neighbors in {elapsed_ms:.2f} ms:")
    for rank, (idx, dist) in enumerate(zip(indices, dists), start=1):
        print(f"  Rank {rank}: sample_id={idx:4d}, distance={dist:.4f}")

    # Build and query Cosine index
    print("\n2. Cosine Distance Index:")
    cos_index = BruteForceIndex(metric="cosine").fit(dataset)
    t0 = time.perf_counter()
    indices_cos, dists_cos = cos_index.query(query_vec, k=5)
    elapsed_ms_cos = (time.perf_counter() - t0) * 1000.0
    print(f"Retrieved top 5 neighbors in {elapsed_ms_cos:.2f} ms:")
    for rank, (idx, dist) in enumerate(zip(indices_cos, dists_cos), start=1):
        print(f"  Rank {rank}: sample_id={idx:4d}, cosine_distance={dist:.4f}")

    print("\nDemo completed successfully.")


def main() -> None:
    """CLI entrypoint."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python main.py [demo]")
        print("\nRun RPForest exact search baseline demonstration.")
        sys.exit(0)

    run_demo()


if __name__ == "__main__":
    main()
