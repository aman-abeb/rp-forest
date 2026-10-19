"""Random Projection Forest index for multi-tree candidate retrieval."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import BinaryIO
import numpy as np

from rpforest.metrics import Metric, pairwise_distances
from rpforest.tree import RPTree


class RPForest:
    """Random Projection Forest approximate nearest neighbor index.

    Combines multiple independently seeded Random Projection Trees to partition
    the vector space. At query time, candidates are retrieved from the matching
    leaf of each tree and pooled for fast exact distance re-ranking.
    """

    def __init__(
        self,
        n_trees: int = 10,
        leaf_size: int = 20,
        max_depth: int | None = None,
        metric: Metric = "euclidean",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the RPForest index.

        Parameters
        ----------
        n_trees : int, default=10
            Number of randomized projection trees to build.
        leaf_size : int, default=20
            Maximum number of points per leaf node in each tree.
        max_depth : int | None, default=None
            Maximum allowable depth for any tree. If None, trees split until leaf_size.
        metric : Metric, default="euclidean"
            Distance metric to use ('euclidean' or 'cosine').
        random_state : int | np.random.Generator | None, default=None
            Deterministic random state seed for reproducible forest construction.
        """
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size}")
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"Unsupported metric '{metric}'. Expected 'euclidean' or 'cosine'.")

        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.metric: Metric = metric
        self.random_state = random_state

        self.trees: list[RPTree] = []
        self._data: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if the forest has been fitted."""
        return len(self.trees) > 0 and self._data is not None

    @property
    def n_samples(self) -> int:
        """Return the number of stored vectors."""
        return 0 if self._data is None else self._data.shape[0]

    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        return 0 if self._data is None else self._data.shape[1]

    @staticmethod
    def _resolve_budget(
        search_budget: int | None,
        max_candidates: int | None,
    ) -> int | None:
        """Validate and resolve search_budget and max_candidates parameters."""
        if search_budget is not None and max_candidates is not None:
            if search_budget != max_candidates:
                raise ValueError(
                    f"Conflicting search budget arguments: search_budget={search_budget} "
                    f"and max_candidates={max_candidates}"
                )
            budget = search_budget
        elif search_budget is not None:
            budget = search_budget
        else:
            budget = max_candidates

        if budget is not None and budget <= 0:
            raise ValueError(f"search_budget / max_candidates must be positive, got {budget}")

        return budget

    def fit(self, X: np.ndarray) -> RPForest:
        """Fit the forest index by building n_trees on data matrix X.

        Parameters
        ----------
        X : np.ndarray
            Input reference vector array of shape (n_samples, n_features).

        Returns
        -------
        RPForest
            The fitted forest instance.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got ndim={arr.ndim}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Cannot fit index with empty array, got shape {arr.shape}")

        self._data = np.ascontiguousarray(arr)
        self.trees.clear()

        # Generate deterministic child seeds for each tree
        master_rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        tree_seeds = master_rng.integers(0, 2**63 - 1, size=self.n_trees)

        for seed in tree_seeds:
            tree = RPTree(
                leaf_size=self.leaf_size,
                max_depth=self.max_depth,
                random_state=seed,
            )
            tree.build(self._data)
            self.trees.append(tree)

        return self

    def get_candidates(
        self,
        q: np.ndarray,
        search_budget: int | None = None,
        max_candidates: int | None = None,
    ) -> np.ndarray:
        """Retrieve candidate sample indices for query vector q across all trees.

        Aggregates indices from the leaf of each tree containing q. If search_budget
        (or max_candidates) is provided, candidates are ranked by their co-occurrence
        frequency across visited leaves and truncated to the budget.

        Parameters
        ----------
        q : np.ndarray
            Query vector of shape (dimension,) or (1, dimension).
        search_budget : int | None, default=None
            Maximum number of candidates to evaluate.
        max_candidates : int | None, default=None
            Synonym for search_budget.

        Returns
        -------
        np.ndarray
            1D array of candidate sample indices.
        """
        if not self.is_fitted or self._data is None:
            raise RuntimeError("RPForest must be fitted before retrieving candidates.")

        query_vec = np.asarray(q, dtype=np.float64).ravel()
        if len(query_vec) != self.dimension:
            raise ValueError(
                f"Query dimension {len(query_vec)} does not match index {self.dimension}"
            )

        budget = self._resolve_budget(search_budget, max_candidates)

        # Collect leaves from all trees
        all_candidates: list[int] = []
        for tree in self.trees:
            leaf_indices = tree.query_leaf(query_vec)
            all_candidates.extend(leaf_indices.tolist())

        if not all_candidates:
            return np.array([], dtype=np.int64)

        # Frequency-weighted candidate ranking
        counts = Counter(all_candidates)
        sorted_by_freq = sorted(counts.keys(), key=lambda idx: (-counts[idx], idx))

        if budget is not None:
            selected = sorted_by_freq[:budget]
            return np.array(selected, dtype=np.int64)

        return np.array(sorted_by_freq, dtype=np.int64)

    def query(
        self,
        q: np.ndarray,
        k: int = 10,
        search_budget: int | None = None,
        max_candidates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the forest for top-k approximate nearest neighbors.

        Parameters
        ----------
        q : np.ndarray
            Query vector of shape (dimension,) or (1, dimension).
        k : int, default=10
            Number of nearest neighbors to retrieve.
        search_budget : int | None, default=None
            Maximum candidate budget cap. Limits exact distance evaluations.
        max_candidates : int | None, default=None
            Synonym for search_budget.

        Returns
        -------
        indices : np.ndarray
            1D array of neighbor indices.
        distances : np.ndarray
            1D array of distances corresponding to each retrieved neighbor.
        """
        if not self.is_fitted or self._data is None:
            raise RuntimeError("RPForest must be fitted before querying.")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        query_vec = np.asarray(q, dtype=np.float64)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        elif query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError(f"query expects 1D or single-row 2D array, got shape {query_vec.shape}")

        if query_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vec.shape[1]} does not match index {self.dimension}"
            )

        budget = self._resolve_budget(search_budget, max_candidates)
        k_actual = min(k, self.n_samples)
        candidates = self.get_candidates(query_vec[0], search_budget=budget)

        # If candidate pool is smaller than k_actual and budget is None or budget >= k_actual,
        # backfill from remaining samples to fulfill k
        max_allowed = k_actual if budget is None else min(k_actual, budget)
        if len(candidates) < max_allowed:
            missing = max_allowed - len(candidates)
            candidate_set = set(candidates.tolist())
            all_indices = np.arange(self.n_samples, dtype=np.int64)
            remaining = [idx for idx in all_indices if idx not in candidate_set]
            candidates = np.concatenate([candidates, remaining[:missing]])

        if len(candidates) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # Compute exact distances exclusively against candidate pool
        cand_vectors = self._data[candidates]
        dists = pairwise_distances(query_vec, cand_vectors, metric=self.metric)[0]

        # Find top k_out among evaluated candidates
        k_out = min(k_actual, len(candidates))
        if len(candidates) <= k_out:
            sort_idx = np.argsort(dists)
        else:
            part_idx = np.argpartition(dists, k_out - 1)[:k_out]
            part_dists = dists[part_idx]
            sort_idx = part_idx[np.argsort(part_dists)]

        return candidates[sort_idx], dists[sort_idx]

    def batch_query(
        self,
        queries: np.ndarray,
        k: int = 10,
        search_budget: int | None = None,
        max_candidates: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the forest with multiple query vectors.

        Parameters
        ----------
        queries : np.ndarray
            Query vectors of shape (n_queries, dimension).
        k : int, default=10
            Number of nearest neighbors to retrieve per query.
        search_budget : int | None, default=None
            Candidate budget cap per query.
        max_candidates : int | None, default=None
            Synonym for search_budget.

        Returns
        -------
        indices : np.ndarray
            2D array of retrieved neighbor indices.
        distances : np.ndarray
            2D array of neighbor distances.
        """
        if not self.is_fitted or self._data is None:
            raise RuntimeError("RPForest must be fitted before querying.")

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

        budget = self._resolve_budget(search_budget, max_candidates)
        n_queries = queries_mat.shape[0]
        k_effective = min(k, self.n_samples) if budget is None else min(k, self.n_samples, budget)

        out_indices = np.empty((n_queries, k_effective), dtype=np.int64)
        out_dists = np.empty((n_queries, k_effective), dtype=np.float64)

        for i in range(n_queries):
            idx, dist = self.query(queries_mat[i], k=k, search_budget=budget)
            out_indices[i] = idx
            out_dists[i] = dist

        return out_indices, out_dists

    def save(self, file_or_path: str | Path | BinaryIO) -> None:
        """Save the fitted forest index to disk or a binary stream.

        Parameters
        ----------
        file_or_path : str | Path | BinaryIO
            Destination file path or writable binary stream.
        """
        from rpforest.serialization import save_forest

        save_forest(self, file_or_path)

    @classmethod
    def load(cls, file_or_path: str | Path | BinaryIO) -> RPForest:
        """Load a fitted RPForest index from disk or a binary stream.

        Parameters
        ----------
        file_or_path : str | Path | BinaryIO
            Source file path or readable binary stream.

        Returns
        -------
        RPForest
            Reconstructed and fitted RPForest index.
        """
        from rpforest.serialization import load_forest

        return load_forest(file_or_path)


# Canonical alias
RandomProjectionForest = RPForest
