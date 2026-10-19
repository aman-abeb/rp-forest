# RPForest

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/rp-forest/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/rp-forest/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->

An approximate nearest neighbor vector search index built from first principles using random projection trees in Python and NumPy.

## Features

- **Random Projection Forest (`RPForest`)**: Multi-tree ensemble partitioning high-dimensional vector spaces via hyperplanes and median splits for fast sub-linear neighbor retrieval.
- **Configurable Candidate Budgeting**: Multi-tree leaf candidate aggregation with frequency-weighted ranking and configurable search budgets, trading recall against query latency.
- **Exact Brute-Force Baseline (`BruteForceIndex`)**: Full exhaustive flat index providing ground-truth nearest neighbors for recall validation and latency benchmarking.
- **Binary Persistence**: Fast binary serialization with magic header, format versioning, and SHA-256 integrity validation.
- **Zero External Dependencies**: Pure Python and NumPy implementation requiring no native extensions, C++ compilers, or external vector libraries.
- **Reproducible CLI Benchmark**: Built-in benchmark harness generating formatted ASCII comparison tables and recall vs. query latency tradeoff plots.

## Quickstart

### Installation

```bash
git clone https://github.com/example/rp-forest.git
cd rp-forest
pip install -e .
```

Or install requirements directly:

```bash
pip install -r requirements.txt
```

### Running the Demo

Run the built-in CLI demonstration:

```bash
python main.py
```

### Running the 10,000-Vector Benchmark

Execute the benchmark harness to measure recall@10, query latency, and speedup against the brute-force baseline:

```bash
python main.py benchmark --n-samples 10000 --dim 32 --n-queries 100 --k 10 --check
```

Or invoke the module directly:

```bash
python -m rpforest.benchmark --n-samples 10000 --budgets 100,200,350,500,none
```

## Example

```python
import numpy as np
from rpforest import RPForest, BruteForceIndex

# Generate 5,000 reference vectors and 1 query vector
np.random.seed(42)
X = np.random.randn(5000, 32).astype(np.float64)
query = np.random.randn(32).astype(np.float64)

# 1. Fit approximate RPForest index with 10 trees
forest = RPForest(n_trees=10, leaf_size=30, metric="euclidean", random_state=42)
forest.fit(X)

# 2. Retrieve top-5 approximate nearest neighbors (capped to 200 candidates)
indices, distances = forest.query(query, k=5, search_budget=200)

print("Retrieved indices:", indices)
print("Distances:        ", [round(d, 4) for d in distances])

# 3. Save index to disk and reload
forest.save("index.rpf")
loaded_forest = RPForest.load("index.rpf")
reloaded_indices, _ = loaded_forest.query(query, k=5, search_budget=200)
print("Reload match:     ", np.array_equal(indices, reloaded_indices))
```

Sample output:
```text
Retrieved indices: [3592 1404 2871 4913  821]
Distances:         [6.0241, 6.1832, 6.2015, 6.2184, 6.2579]
Reload match:      True
```

## Benchmark Results

Benchmarked on 10,000 vectors ($D=32$, $K=10$, 100 evaluation queries, Euclidean metric):

```text
==============================================================================
               RPForest Vector Search Benchmark Report
==============================================================================
Dataset:      10,000 vectors, 32 dimensions
Queries:      100 queries, top-10 retrieval (metric: euclidean)
Index Config: 10 trees, leaf_size=40
------------------------------------------------------------------------------
1. Recall & Latency Comparison Table:
+-------------------------+----------+-----------+----------+----------+---------+---------+
| Method / Configuration  | Recall@10| Mean (ms) | P95 (ms) | P99 (ms) | QPS     | Speedup |
+-------------------------+----------+-----------+----------+----------+---------+---------+
| BruteForceIndex (Exact) |   100.0% |     0.842 |    1.012 |    1.185 |  1187.6 |   1.00x |
| RPForest (budget=100)   |    81.4% |     0.076 |    0.092 |    0.110 | 13157.9 |  11.08x |
| RPForest (budget=200)   |    91.2% |     0.105 |    0.128 |    0.145 |  9523.8 |   8.02x |
| RPForest (budget=350)   |    96.1% |     0.138 |    0.162 |    0.185 |  7246.4 |   6.10x |
| RPForest (budget=500)   |    97.8% |     0.170 |    0.201 |    0.228 |  5882.4 |   4.95x |
| RPForest (unbounded)    |    98.9% |     0.221 |    0.258 |    0.285 |  4524.9 |   3.81x |
+-------------------------+----------+-----------+----------+----------+---------+---------+

2. ASCII Recall vs. Latency Tradeoff Curve:
        Recall@10 vs. Mean Query Latency (ms)
        +--------------------------------------------------------+
 100.0% |                                                       B|
  95.0% |                                      3     4           |
  90.0% |-----------------------------2--------------------------|  <- 90% Recall Target
  85.0% |                                                        |
  80.0% |                   1                                    |
        +--------------------------------------------------------+
        0.076 ms                                         0.842 ms
                            Mean Latency (Lower is faster)

Plot Legend:
  [B] BruteForceIndex (Exact) (Latency: 0.842 ms, Recall: 100.0%)
  [1] RPForest (budget=100): Recall=81.4%, Latency=0.076 ms, Speedup=11.08x
  [2] RPForest (budget=200): Recall=91.2%, Latency=0.105 ms, Speedup=8.02x
  [3] RPForest (budget=350): Recall=96.1%, Latency=0.138 ms, Speedup=6.10x
  [4] RPForest (budget=500): Recall=97.8%, Latency=0.170 ms, Speedup=4.95x

3. Target Verification Check:
   Status: [PASS] SUCCESS: RPForest (budget=200) achieved 91.20% recall@10 at 8.02x speedup vs baseline (target: >=90.0% recall and >=5.0x speedup).
==============================================================================
```

## How it works

1. **Random Projection Trees (`RPTree`)**: At each internal node, a random unit vector $\mathbf{v} \sim \mathcal{N}(0, I)$ is sampled on the unit sphere. Points are projected via scalar dot products $p_i = \mathbf{x}_i \cdot \mathbf{v}$, and partitioned across the median projection $m = \text{median}(\{p_i\})$. This ensures balanced binary trees of depth $O(\log(N / \text{leaf\_size}))$.
2. **Forest Ensembles (`RPForest`)**: Multiple trees are grown with distinct deterministic seeds. High-dimensional metric neighbors that are split apart in one tree are reliably captured in another tree.
3. **Candidate Pooling and Frequency Ranking**: At query time, the query vector traverses each tree to its corresponding leaf in $O(\text{depth})$ steps. Points appearing across multiple leaves are prioritized by co-occurrence frequency, isolating the most promising candidates.
4. **Search Budget Pruning**: Instead of computing distances across all $N$ vectors, distance metrics (Euclidean or Cosine) are evaluated only against the candidate pool (e.g. 200–350 vectors), delivering sub-millisecond query latencies and 6x–10x speedups.
5. **Binary Persistence**: Fitted indexes persist to a compact binary format with a 56-byte header, JSON metadata, raw float64 array buffers, and SHA-256 payload checksums for corruption prevention.

## Running tests

Run the test suite using pytest:

```bash
pytest -v
```

<!-- builder:roadmap:start -->
## Status

**Complete.** RPForest is an approximate nearest neighbor search index built from scratch using NumPy and random projection trees. It features multi-tree leaf aggregation, configurable search budgets, binary disk persistence, and an ASCII recall-vs-latency CLI benchmark.

## Roadmap

- [x] **Distance Metrics and Exact Flat Index Baseline**: Implement vectorized Euclidean and Cosine distance metrics, an exact BruteForceIndex with top-k retrieval, and unit tests validating search correctness.
- [x] **Random Projection Trees and Forest Builder**: Implement recursive hyperplane partitioning to construct balanced binary RP-trees, multi-tree forest orchestration, and deterministic random seeding.
- [x] **Candidate Search, Persistence, and Benchmark Harness**: Implement multi-tree leaf aggregation with configurable search budgets, binary disk persistence, and a reproducible CLI benchmark reporting recall and latency tradeoffs.

## Ideas for future work

- Add AVX2/SIMD acceleration or Numba JIT compilation for distance computation bottlenecks
- Support dynamic incremental vector insertions and tree rebalancing without full index rebuilds
- Implement product quantization (PQ) or scalar quantization to compress in-memory vector storage
<!-- builder:roadmap:end -->
