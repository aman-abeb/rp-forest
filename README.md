# RPForest

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/rp-forest/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/rp-forest/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->


An approximate nearest neighbor vector search index built from first principles using random projection trees in Python and NumPy.

## Features

- **Vectorized Distance Metrics**: Fast, numerically stable Euclidean and Cosine pairwise distances leveraging NumPy matrix operations.
- **Exact Brute-Force Baseline**: Complete `BruteForceIndex` baseline providing ground-truth nearest neighbors for recall and latency evaluation.
- **Zero External Dependencies**: Pure Python and NumPy implementation requiring no heavy native extensions or C++ toolchains.
- **High Precision**: Built-in edge-case safety for zero-norm vectors, small floating-point roundoffs, and variable batch sizes.

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

Run the built-in CLI demo to search across 1,000 synthetic vectors:

```bash
python main.py
```

## Example

```python
import numpy as np
from rpforest.exact import BruteForceIndex

# Generate reference dataset and query vector
np.random.seed(42)
X = np.random.randn(1000, 32).astype(np.float32)
query = np.random.randn(32).astype(np.float32)

# Fit exact flat index with Euclidean distance
index = BruteForceIndex(metric="euclidean").fit(X)

# Retrieve top 5 nearest neighbors
indices, distances = index.query(query, k=5)
print("Nearest neighbor indices:", indices)
print("Distances:", distances)
```

Sample output:
```text
Nearest neighbor indices: [733 462 250 812 607]
Distances: [6.1342 6.2709 6.2954 6.3012 6.3275]
```

## How it works

RPForest builds high-performance vector search indexes based on Random Projection Trees (RP-Trees):

1. **Distance Computation**: Pairwise Euclidean and Cosine distances are formulated as matrix multiplications $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 x \cdot y^T$, taking full advantage of optimized BLAS routines.
2. **Flat Ground Truth (`BruteForceIndex`)**: Performs exact distance calculation against all indexed points, using partial sorting (`argpartition`) to extract the top-$k$ nearest neighbors in $O(N)$ expected time.
3. **Random Projection Partitioning (Coming in Milestones 2 & 3)**: High-dimensional space is recursively partitioned using hyperplanes with random normal unit vectors, grouping metric neighbors into leaf bins.

## Running tests

Run the test suite using pytest:

```bash
pytest -v
```

<!-- builder:roadmap:start -->
## Status

**In active development:** 0 of 3 milestones done.

## Roadmap

- [ ] **Distance Metrics and Exact Flat Index Baseline**: Implement vectorized Euclidean and Cosine distance metrics, an exact BruteForceIndex with top-k retrieval, and unit tests validating search correctness.
- [ ] **Random Projection Trees and Forest Builder**: Implement recursive hyperplane partitioning to construct balanced binary RP-trees, multi-tree forest orchestration, and deterministic random seeding.
- [ ] **Candidate Search, Persistence, and Benchmark Harness**: Implement multi-tree leaf aggregation with configurable search budgets, binary disk persistence, and a reproducible CLI benchmark reporting recall and latency tradeoffs.
<!-- builder:roadmap:end -->
