"""Vectorized distance metric implementations for vector search."""

from __future__ import annotations

from typing import Literal
import numpy as np

Metric = Literal["euclidean", "cosine"]


def _ensure_2d(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate and convert 1D/2D arrays into 2D float64 numpy arrays."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float64)
    else:
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}")

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty, got shape {arr.shape}")

    return arr


def euclidean_distances(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    squared: bool = False,
) -> np.ndarray:
    """Compute vectorized pairwise Euclidean distances between vectors.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n_samples_X, n_features) or 1D array of shape (n_features,).
    Y : np.ndarray | None
        Optional matrix of shape (n_samples_Y, n_features) or 1D array. If None, Y = X.
    squared : bool, default=False
        If True, return squared Euclidean distances ||x - y||^2.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples_X, n_samples_Y).
    """
    X_mat = _ensure_2d(X, "X")
    if Y is None:
        Y_mat = X_mat
    else:
        Y_mat = _ensure_2d(Y, "Y")

    if X_mat.shape[1] != Y_mat.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X has {X_mat.shape[1]} features, "
            f"but Y has {Y_mat.shape[1]} features."
        )

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>
    x_sq = np.sum(X_mat * X_mat, axis=1, keepdims=True)
    y_sq = np.sum(Y_mat * Y_mat, axis=1, keepdims=True).T
    xy = np.dot(X_mat, Y_mat.T)

    dist_sq = x_sq + y_sq - 2.0 * xy
    # Guard against negative values due to floating point roundoff
    np.maximum(dist_sq, 0.0, out=dist_sq)

    if Y is None:
        # Ensure identical points on diagonal evaluate strictly to 0.0
        np.fill_diagonal(dist_sq, 0.0)

    if squared:
        return dist_sq
    return np.sqrt(dist_sq)


def cosine_distances(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute vectorized pairwise Cosine distances between vectors.

    Cosine distance is defined as: 1 - cosine_similarity(u, v).
    Values are bounded in [0.0, 2.0]. When encountering a zero-norm vector,
    similarity is treated as 0.0 (yielding distance 1.0), unless both vectors
    are identically zero, in which case distance is 0.0.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n_samples_X, n_features) or 1D array of shape (n_features,).
    Y : np.ndarray | None
        Optional matrix of shape (n_samples_Y, n_features) or 1D array. If None, Y = X.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples_X, n_samples_Y).
    """
    X_mat = _ensure_2d(X, "X")
    if Y is None:
        Y_mat = X_mat
    else:
        Y_mat = _ensure_2d(Y, "Y")

    if X_mat.shape[1] != Y_mat.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X has {X_mat.shape[1]} features, "
            f"but Y has {Y_mat.shape[1]} features."
        )

    x_norm = np.linalg.norm(X_mat, axis=1, keepdims=True)
    y_norm = np.linalg.norm(Y_mat, axis=1, keepdims=True)

    # Identify zero-norm vectors
    eps = 1e-15
    x_is_zero = x_norm.ravel() < eps
    y_is_zero = y_norm.ravel() < eps

    # Replace 0 norms with 1.0 to prevent division by zero in normalization
    safe_x_norm = np.where(x_norm < eps, 1.0, x_norm)
    safe_y_norm = np.where(y_norm < eps, 1.0, y_norm)

    X_unit = X_mat / safe_x_norm
    Y_unit = Y_mat / safe_y_norm

    similarity = np.dot(X_unit, Y_unit.T)
    # Clip numerical inaccuracies outside [-1.0, 1.0]
    np.clip(similarity, -1.0, 1.0, out=similarity)

    dist = 1.0 - similarity

    # Adjust zero vectors: distance is 1.0 if one is zero, 0.0 if both are zero
    if np.any(x_is_zero) or np.any(y_is_zero):
        mask_x = x_is_zero[:, None]
        mask_y = y_is_zero[None, :]
        either_zero = mask_x | mask_y
        both_zero = mask_x & mask_y
        dist[either_zero] = 1.0
        dist[both_zero] = 0.0

    if Y is None:
        np.fill_diagonal(dist, 0.0)

    np.clip(dist, 0.0, 2.0, out=dist)
    return dist


def pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    metric: Metric = "euclidean",
) -> np.ndarray:
    """Compute pairwise distances for a supported metric.

    Parameters
    ----------
    X : np.ndarray
        First set of vectors.
    Y : np.ndarray | None
        Second set of vectors. If None, Y = X.
    metric : Metric, default="euclidean"
        Distance metric: 'euclidean' or 'cosine'.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix.
    """
    if metric == "euclidean":
        return euclidean_distances(X, Y)
    elif metric == "cosine":
        return cosine_distances(X, Y)
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported metrics are 'euclidean' and 'cosine'."
        )
