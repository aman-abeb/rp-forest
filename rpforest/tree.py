"""Random Projection Tree implementation for recursive space partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class RPNode:
    """Node in a Random Projection Tree.

    Parameters
    ----------
    is_leaf : bool
        True if the node is a leaf, False for internal decision nodes.
    indices : np.ndarray | None
        1D array of sample indices stored at this leaf node (None for internal nodes).
    hyperplane : np.ndarray | None
        Normalized random projection unit vector (dim,) for internal nodes.
    split_val : float | None
        Scalar threshold splitting data along the projection hyperplane.
    left : RPNode | None
        Left child node (projections <= split_val).
    right : RPNode | None
        Right child node (projections > split_val).
    depth : int
        Depth of this node from the root (root has depth 0).
    """

    is_leaf: bool
    indices: np.ndarray | None = None
    hyperplane: np.ndarray | None = None
    split_val: float | None = None
    left: RPNode | None = None
    right: RPNode | None = None
    depth: int = 0

    @property
    def n_points(self) -> int:
        """Number of points stored in this node if it is a leaf, otherwise 0."""
        if self.indices is not None:
            return len(self.indices)
        return 0


class RPTree:
    """Random Projection Tree for balanced binary spatial partitioning.

    Recursively partitions vectors using hyperplanes defined by random unit vectors
    and splits at the median of point projections.
    """

    def __init__(
        self,
        leaf_size: int = 10,
        max_depth: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the Random Projection Tree.

        Parameters
        ----------
        leaf_size : int, default=10
            Maximum number of data points permitted in a leaf node before splitting.
        max_depth : int | None, default=None
            Maximum allowable depth for the tree. If None, depth is bounded only by leaf_size.
        random_state : int | np.random.Generator | None, default=None
            Seed or random number generator for deterministic hyperplane sampling.
        """
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size}")
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")

        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        self.root: RPNode | None = None
        self._n_samples: int = 0
        self._dimension: int = 0

    @property
    def is_fitted(self) -> bool:
        """Return True if tree has been built."""
        return self.root is not None

    @property
    def n_samples(self) -> int:
        """Total number of indexed vectors."""
        return self._n_samples

    @property
    def dimension(self) -> int:
        """Dimensionality of indexed vectors."""
        return self._dimension

    def build(self, X: np.ndarray, indices: np.ndarray | None = None) -> RPTree:
        """Build the random projection tree from data matrix X.

        Parameters
        ----------
        X : np.ndarray
            Reference data array of shape (n_samples, n_features).
        indices : np.ndarray | None, default=None
            Optional subset of sample indices to build tree over. If None, uses all rows of X.

        Returns
        -------
        RPTree
            The fitted tree instance.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got ndim={arr.ndim}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Cannot build tree from empty array with shape {arr.shape}")

        self._n_samples = arr.shape[0]
        self._dimension = arr.shape[1]

        if indices is None:
            node_indices = np.arange(self._n_samples, dtype=np.int64)
        else:
            node_indices = np.asarray(indices, dtype=np.int64)

        self.root = self._build_recursive(arr, node_indices, depth=0)
        return self

    def _generate_hyperplane(self, dim: int) -> np.ndarray:
        """Sample a normalized random direction vector uniformly on the unit sphere."""
        vec = self._rng.normal(0.0, 1.0, size=dim)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            # Fallback to standard basis unit vector in case of zero vector
            vec = np.zeros(dim, dtype=np.float64)
            vec[0] = 1.0
            return vec
        return vec / norm

    def _build_recursive(self, X: np.ndarray, indices: np.ndarray, depth: int) -> RPNode:
        """Recursively construct tree nodes via median hyperplane splits."""
        n_pts = len(indices)

        # Base condition: leaf size threshold or max depth reached
        if n_pts <= self.leaf_size or (self.max_depth is not None and depth >= self.max_depth):
            return RPNode(is_leaf=True, indices=indices, depth=depth)

        # Attempt to split points along random hyperplanes
        dim = X.shape[1]
        split_found = False
        hyperplane: np.ndarray | None = None
        split_val: float | None = None
        left_idx: np.ndarray | None = None
        right_idx: np.ndarray | None = None

        for _ in range(5):
            v = self._generate_hyperplane(dim)
            pts = X[indices]
            projections = np.dot(pts, v)
            p_min, p_max = float(np.min(projections)), float(np.max(projections))

            # If all projections are nearly identical, retry with another hyperplane
            if (p_max - p_min) < 1e-12:
                continue

            median_val = float(np.median(projections))
            left_mask = projections <= median_val
            right_mask = ~left_mask

            cand_left = indices[left_mask]
            cand_right = indices[right_mask]

            # Ensure neither child is empty
            if len(cand_left) > 0 and len(cand_right) > 0:
                hyperplane = v
                split_val = median_val
                left_idx = cand_left
                right_idx = cand_right
                split_found = True
                break

        # If no valid hyperplane split was found (e.g. collinear or duplicate points),
        # perform a deterministic index-half partition if still above leaf_size,
        # or return a leaf.
        if not split_found or left_idx is None or right_idx is None:
            # Check if all points in subset are identical
            subset = X[indices]
            if np.allclose(subset, subset[0:1], atol=1e-12):
                return RPNode(is_leaf=True, indices=indices, depth=depth)

            # Arbitrary equal split to guarantee recursive progress
            mid = n_pts // 2
            left_idx = indices[:mid]
            right_idx = indices[mid:]
            # Use a dummy unit vector orthogonal to axis 0
            hyperplane = np.zeros(dim, dtype=np.float64)
            hyperplane[0] = 1.0
            split_val = 0.0

        left_node = self._build_recursive(X, left_idx, depth=depth + 1)
        right_node = self._build_recursive(X, right_idx, depth=depth + 1)

        return RPNode(
            is_leaf=False,
            hyperplane=hyperplane,
            split_val=split_val,
            left=left_node,
            right=right_node,
            depth=depth,
        )

    def query_leaf(self, q: np.ndarray) -> np.ndarray:
        """Traverse the tree to locate the leaf node indices for query vector q.

        Parameters
        ----------
        q : np.ndarray
            Query vector of shape (dimension,) or (1, dimension).

        Returns
        -------
        np.ndarray
            1D array of sample indices stored in the traversed leaf.
        """
        if not self.is_fitted or self.root is None:
            raise RuntimeError("RPTree must be built before querying.")

        query_vec = np.asarray(q, dtype=np.float64).ravel()
        if len(query_vec) != self._dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vec)} does not match tree dimension {self._dimension}"
            )

        curr = self.root
        while not curr.is_leaf:
            if np.dot(query_vec, curr.hyperplane) <= curr.split_val:
                curr = curr.left
            else:
                curr = curr.right

        return curr.indices

    def get_all_leaves(self) -> list[RPNode]:
        """Collect all leaf nodes in the tree."""
        if self.root is None:
            return []

        leaves: list[RPNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
            else:
                if node.right is not None:
                    stack.append(node.right)
                if node.left is not None:
                    stack.append(node.left)
        return leaves

    @property
    def n_leaves(self) -> int:
        """Total number of leaf nodes in the tree."""
        return len(self.get_all_leaves())

    @property
    def max_tree_depth(self) -> int:
        """Maximum depth achieved across all leaf nodes in the tree."""
        leaves = self.get_all_leaves()
        return max((leaf.depth for leaf in leaves), default=0)
