"""RPForest: Approximate Nearest Neighbors with Random Projection Trees."""

from rpforest.exact import BruteForceIndex
from rpforest.metrics import (
    Metric,
    cosine_distances,
    euclidean_distances,
    pairwise_distances,
)

__version__ = "0.1.0"
__all__ = [
    "BruteForceIndex",
    "Metric",
    "cosine_distances",
    "euclidean_distances",
    "pairwise_distances",
]
