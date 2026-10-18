"""RPForest: Approximate Nearest Neighbors with Random Projection Trees."""

from rpforest.exact import BruteForceIndex
from rpforest.forest import RPForest, RandomProjectionForest
from rpforest.metrics import (
    Metric,
    cosine_distances,
    euclidean_distances,
    pairwise_distances,
)
from rpforest.tree import RPNode, RPTree

__version__ = "0.1.0"
__all__ = [
    "BruteForceIndex",
    "Metric",
    "RPForest",
    "RPNode",
    "RPTree",
    "RandomProjectionForest",
    "cosine_distances",
    "euclidean_distances",
    "pairwise_distances",
]
