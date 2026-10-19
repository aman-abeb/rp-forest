"""RPForest: Approximate Nearest Neighbors with Random Projection Trees."""

from rpforest.benchmark import (
    BenchmarkRunResult,
    BenchmarkSuiteResult,
    calculate_recall_at_k,
    generate_dataset,
    run_benchmark,
)
from rpforest.exact import BruteForceIndex
from rpforest.forest import RPForest, RandomProjectionForest
from rpforest.metrics import (
    Metric,
    cosine_distances,
    euclidean_distances,
    pairwise_distances,
)
from rpforest.serialization import load_forest, save_forest
from rpforest.tree import RPNode, RPTree

__version__ = "0.1.0"
__all__ = [
    "BenchmarkRunResult",
    "BenchmarkSuiteResult",
    "BruteForceIndex",
    "Metric",
    "RPForest",
    "RPNode",
    "RPTree",
    "RandomProjectionForest",
    "calculate_recall_at_k",
    "cosine_distances",
    "euclidean_distances",
    "generate_dataset",
    "load_forest",
    "pairwise_distances",
    "run_benchmark",
    "save_forest",
]
