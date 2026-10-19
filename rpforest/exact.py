"""Exact brute-force nearest neighbor search baseline."""

from __future__ import annotations

import numpy as np
from rpforest.metrics import Metric


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

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

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

        qv = query_mat[0]
        if self.metric == "euclidean":
            diff = self._data - qv
            dists = np.sqrt(np.sum(diff * diff, axis=1))
        elif self.metric == "cosine":
            q_norm = float(np.linalg.norm(qv))
            data_norms = np.linalg.norm(self._data, axis=1)
            if q_norm < 1e-15:
                dists = np.where(data_norms < 1e-15, 0.0, 1.0)
            else:
                safe_data = np.where(data_norms < 1e-15, 1.0, data_norms)
                sim = np.dot(self._data, qv) / (safe_data * q_norm)
                np.clip(sim, -1.0, 1.0, out=sim)
                dists = 1.0 - sim
                dists[data_norms < 1e-15] = 1.0
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'.")

        sorted_idx = np.argsort(dists)
        k_actual = min(k, self.n_samples)
        top_k_idx = sorted_idx[:k_actual]
        return top_k_idx, dists[top_k_idx]

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

        n_queries = queries_mat.shape[0]
        k_actual = min(k, self.n_samples)

        out_indices = np.empty((n_queries, k_actual), dtype=np.int64)
        out_dists = np.empty((n_queries, k_actual), dtype=np.float64)

        for i in range(n_queries):
            idx, dst = self.query(queries_mat[i], k=k)
            out_indices[i] = idx
            out_dists[i] = dst

        return out_indices, out_dists
