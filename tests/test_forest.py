"""Unit tests for RPForest multi-tree candidate search and indexing."""

import numpy as np
import pytest
from rpforest.exact import BruteForceIndex
from rpforest.forest import RPForest, RandomProjectionForest


def test_forest_initialization_validation():
    """Verify parameter validation in forest constructor."""
    with pytest.raises(ValueError, match="n_trees must be >= 1"):
        RPForest(n_trees=0)
    with pytest.raises(ValueError, match="leaf_size must be >= 1"):
        RPForest(leaf_size=0)
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        RPForest(max_depth=-1)
    with pytest.raises(ValueError, match="Unsupported metric"):
        RPForest(metric="invalid")  # type: ignore


def test_forest_alias():
    """Verify RandomProjectionForest is an alias for RPForest."""
    assert RandomProjectionForest is RPForest


def test_forest_unfitted_raises():
    """Verify operations on unfitted forest raise RuntimeError."""
    forest = RPForest()
    with pytest.raises(RuntimeError, match="must be fitted"):
        forest.query(np.array([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="must be fitted"):
        forest.get_candidates(np.array([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="must be fitted"):
        forest.batch_query(np.array([[1.0, 2.0]]))


def test_forest_fit_and_properties():
    """Verify forest fit properties and tree count."""
    np.random.seed(42)
    X = np.random.randn(200, 16)
    forest = RPForest(n_trees=8, leaf_size=25, random_state=42).fit(X)

    assert forest.is_fitted
    assert forest.n_samples == 200
    assert forest.dimension == 16
    assert len(forest.trees) == 8
    for tree in forest.trees:
        assert tree.is_fitted
        assert tree.dimension == 16


def test_forest_deterministic_seeding():
    """Two forests built with identical seeds produce identical candidate sets and rankings."""
    np.random.seed(123)
    X = np.random.randn(150, 10)
    q = np.random.randn(10)

    forest1 = RPForest(n_trees=6, leaf_size=15, random_state=42).fit(X)
    forest2 = RPForest(n_trees=6, leaf_size=15, random_state=42).fit(X)

    cand1 = forest1.get_candidates(q)
    cand2 = forest2.get_candidates(q)
    assert np.array_equal(cand1, cand2)

    idx1, d1 = forest1.query(q, k=5)
    idx2, d2 = forest2.query(q, k=5)
    assert np.array_equal(idx1, idx2)
    assert np.allclose(d1, d2)


def test_forest_search_budget_caps_candidates():
    """Verify candidate budget caps the candidate pool."""
    np.random.seed(42)
    X = np.random.randn(300, 8)
    forest = RPForest(n_trees=10, leaf_size=30, random_state=42).fit(X)
    q = np.random.randn(8)

    full_cands = forest.get_candidates(q, search_budget=None)
    budget_cands = forest.get_candidates(q, search_budget=20)

    assert len(budget_cands) <= 20
    assert len(full_cands) >= len(budget_cands)
    # Budget candidates must be a subset of full candidates
    assert set(budget_cands.tolist()).issubset(set(full_cands.tolist()))


def test_forest_query_topk_ranking_and_self_retrieval():
    """Querying an indexed point returns itself as the closest candidate."""
    np.random.seed(999)
    X = np.random.randn(100, 12)
    forest = RPForest(n_trees=10, leaf_size=15, random_state=999).fit(X)

    # Query existing point X[5]
    indices, dists = forest.query(X[5], k=5)
    assert indices[0] == 5
    assert np.isclose(dists[0], 0.0, atol=1e-7)
    # Distances must be sorted non-decreasingly
    assert np.all(np.diff(dists) >= -1e-12)


def test_forest_cosine_metric():
    """Verify RPForest functions correctly with cosine metric."""
    np.random.seed(77)
    X = np.random.randn(80, 8)
    forest = RPForest(n_trees=8, leaf_size=15, metric="cosine", random_state=77).fit(X)
    q = np.random.randn(8)
    indices, dists = forest.query(q, k=5)
    assert len(indices) == 5
    assert len(dists) == 5
    assert np.all(dists >= 0.0)
    assert np.all(dists <= 2.0)


def test_forest_high_recall_vs_bruteforce():
    """RPForest achieves strong recall against BruteForceIndex ground truth."""
    np.random.seed(42)
    n_samples, dim = 500, 16
    X = np.random.randn(n_samples, dim).astype(np.float64)
    queries = np.random.randn(20, dim).astype(np.float64)

    exact_index = BruteForceIndex(metric="euclidean").fit(X)
    forest = RPForest(n_trees=15, leaf_size=20, metric="euclidean", random_state=42).fit(X)

    k = 10
    recalls = []
    for q in queries:
        true_idx, _ = exact_index.query(q, k=k)
        approx_idx, _ = forest.query(q, k=k)
        overlap = len(set(true_idx).intersection(set(approx_idx)))
        recalls.append(overlap / k)

    avg_recall = float(np.mean(recalls))
    # Forest with 15 trees should achieve >= 0.80 recall on 500 points
    assert avg_recall >= 0.80, f"Average recall {avg_recall:.3f} was below expected threshold"


def test_forest_batch_query():
    """Verify batch_query matches sequential query calls."""
    np.random.seed(33)
    X = np.random.randn(100, 8)
    forest = RPForest(n_trees=5, leaf_size=15, random_state=33).fit(X)
    queries = np.random.randn(5, 8)

    batch_idx, batch_dists = forest.batch_query(queries, k=4)
    assert batch_idx.shape == (5, 4)
    assert batch_dists.shape == (5, 4)

    for i in range(5):
        single_idx, single_dists = forest.query(queries[i], k=4)
        assert np.array_equal(batch_idx[i], single_idx)
        assert np.allclose(batch_dists[i], single_dists)


def test_forest_k_greater_than_n_samples():
    """Requesting k > n_samples gracefully returns all samples."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    forest = RPForest(n_trees=3, leaf_size=1).fit(X)
    indices, dists = forest.query(np.array([2.0, 3.0]), k=10)
    assert len(indices) == 2
    assert len(dists) == 2
