"""Unit tests for BruteForceIndex exact retrieval baseline."""

import numpy as np
import pytest
from rpforest.exact import BruteForceIndex


def test_unfitted_index_raises():
    """Verify querying an unfitted index raises RuntimeError."""
    idx = BruteForceIndex()
    with pytest.raises(RuntimeError, match="Index must be fitted"):
        idx.query(np.array([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="Index must be fitted"):
        idx.batch_query(np.array([[1.0, 2.0]]))


def test_invalid_metric():
    """Verify unsupported metric raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        BruteForceIndex(metric="hamming")  # type: ignore


def test_exact_self_retrieval():
    """Querying an existing data point returns itself as the rank-1 neighbor with distance ~0."""
    np.random.seed(123)
    n_samples, dim = 50, 8
    data = np.random.randn(n_samples, dim)

    for metric in ("euclidean", "cosine"):
        index = BruteForceIndex(metric=metric).fit(data)
        assert index.n_samples == n_samples
        assert index.dimension == dim

        # Query sample 7
        query_point = data[7]
        indices, dists = index.query(query_point, k=5)
        assert indices[0] == 7
        assert np.isclose(dists[0], 0.0, atol=1e-7)


def test_batch_query_shape_and_ranking():
    """Verify batch query returns expected shapes and sorted distances."""
    np.random.seed(42)
    data = np.random.randn(100, 16)
    queries = np.random.randn(10, 16)
    k = 7

    index = BruteForceIndex(metric="euclidean").fit(data)
    indices, dists = index.batch_query(queries, k=k)

    assert indices.shape == (10, k)
    assert dists.shape == (10, k)

    # Distances must be strictly non-decreasing across columns
    for i in range(10):
        assert np.all(np.diff(dists[i]) >= -1e-12)


def test_k_greater_than_n_samples():
    """Verify requesting k > n_samples gracefully clamps k to n_samples."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = BruteForceIndex().fit(data)
    indices, dists = index.query(np.array([2.0, 3.0]), k=10)
    assert len(indices) == 3
    assert len(dists) == 3


def test_single_vector_dataset():
    """Verify index operates correctly with a single vector."""
    data = np.array([[10.0, 20.0]])
    index = BruteForceIndex().fit(data)
    indices, dists = index.query(np.array([10.0, 21.0]), k=1)
    assert indices[0] == 0
    assert np.isclose(dists[0], 1.0)


def test_query_dimension_mismatch():
    """Verify querying with wrong vector dimension raises ValueError."""
    data = np.random.randn(20, 5)
    index = BruteForceIndex().fit(data)
    with pytest.raises(ValueError, match="dimension .* does not match"):
        index.query(np.array([1.0, 2.0, 3.0]))


def test_negative_or_zero_k():
    """Verify invalid k raises ValueError."""
    data = np.random.randn(10, 4)
    index = BruteForceIndex().fit(data)
    with pytest.raises(ValueError, match="k must be positive"):
        index.query(data[0], k=0)
