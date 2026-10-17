"""Exact brute-force nearest neighbor search baseline."""

from __future__ import annotations

import numpy as np
from rpforest.metrics import Metric, pairwise_distances


class BruteForceIndex:
    """Exhaustive flat index performing exact k-nearest neighbor retrieval.

    Serves as the ground truth baseline to compute recall and benchmark search speed
    for approximate nearest neighbor algorithms.
    """

    def __init__(self, metric: Metric = "euclidean") -> None:
        """Initialize the brute force index.

        Parameters
        ----------
        metric : Metric, default="euclidean"
            Distance metric to use for retrieval ('euclidean' or 'cosine').
        """
        if metric not in ("euclidean", "cosine"):
            raise ValueError(
                f"Unsupported metric '{metric}'. Expected 'euclidean' or 'cosine'."
            )
        self.metric: Metric = metric
        self._data: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if index has been populated with vectors."""
        return self._data is not None

    @property
    def n_samples(self) -> int:
        """Return the number of vectors stored in the index."""
        return 0 if self._data is None else self._data.shape[0]

    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        return 0 if self._data is None else self._data.shape[1]

    def fit(self, X: np.ndarray) -> BruteForceIndex:
        """Populate index with reference vectors.

        Parameters
        ----------
        X : np.ndarray
            Reference vectors of shape (n_samples, n_features).

        Returns
        -------
        BruteForceIndex
            The fitted index instance.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got ndim={arr.ndim}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Cannot fit index with empty array, got shape {arr.shape}")

        self._data = np.ascontiguousarray(arr)
        return self

    def query(
        self,
        q: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the index for the top-k nearest neighbors to vector q.

        Parameters
        ----------
        q : np.ndarray
            Query vector of shape (n_features,) or (1, n_features).
        k : int, default=10
            Number of nearest neighbors to retrieve.

        Returns
        -------
        indices : np.ndarray
            1D array of indices of the nearest neighbors in ascending order of distance.
        distances : np.ndarray
            1D array of distances corresponding to each retrieved neighbor.
        """
        if not self.is_fitted or self._data is None:
            raise RuntimeError("Index must be fitted before querying.")

        query_mat = np.asarray(q, dtype=np.float64)
        if query_mat.ndim == 1:
            query_mat = query_mat.reshape(1, -1)
        elif query_mat.ndim != 2 or query_mat.shape[0] != 1:
            raise ValueError(
                f"query expects 1D vector or single-row 2D matrix, got shape {query_mat.shape}"
            )

        if query_mat.shape[1] != self.dimension:
            raise ValueError(
                f"Query feature dimension {query_mat.shape[1]} does not match index {self.dimension}"
            )

        indices_batch, dists_batch = self.batch_query(query_mat, k=k)
        return indices_batch[0], dists_batch[0]

    def batch_query(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the index with multiple query vectors.

        Parameters
        ----------
        queries : np.ndarray
            Query vectors of shape (n_queries, n_features).
        k : int, default=10
            Number of nearest neighbors to retrieve per query.

        Returns
        -------
        indices : np.ndarray
            2D array of shape (n_queries, k_actual) containing neighbor indices.
        distances : np.ndarray
            2D array of shape (n_queries, k_actual) containing distances.
        """
        if not self.is_fitted or self._data is None:
            raise RuntimeError("Index must be fitted before querying.")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        queries_mat = np.asarray(queries, dtype=np.float64)
        if queries_mat.ndim == 1:
            queries_mat = queries_mat.reshape(1, -1)
        elif queries_mat.ndim != 2:
            raise ValueError(f"queries must be 1D or 2D array, got ndim={queries_mat.ndim}")

        if queries_mat.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {queries_mat.shape[1]} does not match index {self.dimension}"
            )

        n_samples = self.n_samples
        k_actual = min(k, n_samples)

        # Compute full distance matrix (n_queries, n_samples)
        dists = pairwise_distances(queries_mat, self._data, metric=self.metric)

        if k_actual == n_samples:
            # Sort all samples
            sorted_idx = np.argsort(dists, axis=1)
        else:
            # Efficient top-k partition then sort the top-k subset
            part_idx = np.argpartition(dists, k_actual - 1, axis=1)[:, :k_actual]
            row_idx = np.arange(queries_mat.shape[0])[:, None]
            part_dists = dists[row_idx, part_idx]
            sort_within_topk = np.argsort(part_dists, axis=1)
            sorted_idx = part_idx[row_idx, sort_within_topk]

        row_idx = np.arange(queries_mat.shape[0])[:, None]
        sorted_dists = dists[row_idx, sorted_idx]

        return sorted_idx, sorted_dists
