"""Unit tests for vectorized distance metrics."""

import numpy as np
import pytest
from rpforest.metrics import cosine_distances, euclidean_distances, pairwise_distances


def test_euclidean_distances_analytical():
    """Verify Euclidean distance matches known hand-calculated values."""
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    dists = euclidean_distances(x)
    assert dists.shape == (2, 2)
    assert np.isclose(dists[0, 0], 0.0)
    assert np.isclose(dists[1, 1], 0.0)
    assert np.isclose(dists[0, 1], 5.0)
    assert np.isclose(dists[1, 0], 5.0)


def test_euclidean_squared():
    """Verify squared Euclidean distance option."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[4.0, 6.0]])
    # (3)^2 + (4)^2 = 25
    dist_sq = euclidean_distances(x, y, squared=True)
    assert np.isclose(dist_sq[0, 0], 25.0)
    dist = euclidean_distances(x, y, squared=False)
    assert np.isclose(dist[0, 0], 5.0)


def test_euclidean_1d_input():
    """Verify 1D input coercion."""
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    d = euclidean_distances(x, y)
    assert d.shape == (1, 1)
    assert np.isclose(d[0, 0], np.sqrt(2.0))


def test_cosine_distances_known_angles():
    """Verify cosine distance for orthogonal, identical, and opposite vectors."""
    x = np.array([
        [1.0, 0.0],   # v0
        [0.0, 1.0],   # v1 (orthogonal to v0)
        [-1.0, 0.0],  # v2 (opposite to v0)
    ])
    dists = cosine_distances(x)
    # Self distance is 0
    assert np.isclose(dists[0, 0], 0.0)
    assert np.isclose(dists[1, 1], 0.0)
    assert np.isclose(dists[2, 2], 0.0)
    # Orthogonal distance is 1.0
    assert np.isclose(dists[0, 1], 1.0)
    assert np.isclose(dists[1, 0], 1.0)
    # Opposite distance is 2.0
    assert np.isclose(dists[0, 2], 2.0)
    assert np.isclose(dists[2, 0], 2.0)


def test_cosine_scale_invariance():
    """Verify cosine distance is invariant to positive scalar multiplication."""
    v1 = np.array([[1.0, 2.0, 3.0]])
    v2 = np.array([[10.0, 20.0, 30.0]])
    d = cosine_distances(v1, v2)
    assert np.isclose(d[0, 0], 0.0, atol=1e-12)


def test_cosine_zero_vector():
    """Verify zero-vector handling does not produce NaN or errors."""
    zero = np.zeros((1, 3))
    normal = np.array([[1.0, 2.0, 3.0]])

    # Distance between zero and normal vector is 1.0
    d_zero_norm = cosine_distances(zero, normal)
    assert not np.isnan(d_zero_norm[0, 0])
    assert np.isclose(d_zero_norm[0, 0], 1.0)

    # Distance between two zero vectors is 0.0
    d_zero_zero = cosine_distances(zero, zero)
    assert not np.isnan(d_zero_zero[0, 0])
    assert np.isclose(d_zero_zero[0, 0], 0.0)


def test_pairwise_distances_wrapper():
    """Test pairwise_distances dispatcher function."""
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    euc = pairwise_distances(X, metric="euclidean")
    cos = pairwise_distances(X, metric="cosine")
    assert euc.shape == (2, 2)
    assert cos.shape == (2, 2)

    with pytest.raises(ValueError, match="Unknown metric"):
        pairwise_distances(X, metric="manhattan")  # type: ignore


def test_dimension_mismatch():
    """Test dimension mismatch error."""
    x = np.ones((2, 3))
    y = np.ones((2, 4))
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        euclidean_distances(x, y)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        cosine_distances(x, y)
