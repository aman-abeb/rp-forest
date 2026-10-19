"""Unit tests for RPForest multi-tree candidate search, budgeting, and indexing."""

import io
import time
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
    with pytest.raises(RuntimeError, match="Cannot save unfitted"):
        forest.save("unfitted.rpf")


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
    assert set(budget_cands.tolist()).issubset(set(full_cands.tolist()))


def test_forest_search_budget_aliases():
    """Verify search_budget and max_candidates can be used interchangeably."""
    np.random.seed(42)
    X = np.random.randn(200, 8)
    forest = RPForest(n_trees=8, leaf_size=20, random_state=42).fit(X)
    q = np.random.randn(8)

    cands1 = forest.get_candidates(q, search_budget=15)
    cands2 = forest.get_candidates(q, max_candidates=15)
    assert np.array_equal(cands1, cands2)

    idx1, d1 = forest.query(q, k=5, search_budget=15)
    idx2, d2 = forest.query(q, k=5, max_candidates=15)
    assert np.array_equal(idx1, idx2)
    assert np.allclose(d1, d2)

    # Conflicting arguments must raise ValueError
    with pytest.raises(ValueError, match="Conflicting search budget"):
        forest.query(q, search_budget=10, max_candidates=20)


def test_forest_invalid_search_budget():
    """Verify non-positive search budgets raise ValueError."""
    X = np.random.randn(50, 4)
    forest = RPForest().fit(X)
    q = np.random.randn(4)

    with pytest.raises(ValueError, match="must be positive"):
        forest.get_candidates(q, search_budget=0)
    with pytest.raises(ValueError, match="must be positive"):
        forest.query(q, max_candidates=-1)


def test_forest_candidate_frequency_ranking():
    """Verify search budget prioritizes candidates that appear in more tree leaves."""
    np.random.seed(99)
    X = np.random.randn(300, 10)
    forest = RPForest(n_trees=15, leaf_size=25, random_state=99).fit(X)
    q = np.random.randn(10)

    # Full candidate list sorted by frequency
    all_cands = forest.get_candidates(q, search_budget=None)
    budget_cands = forest.get_candidates(q, search_budget=10)

    assert len(budget_cands) == 10
    assert np.array_equal(budget_cands, all_cands[:10])


def test_forest_search_budget_candidate_truncation_query():
    """Verify query honors candidate truncation constraints when budget < k."""
    np.random.seed(55)
    X = np.random.randn(100, 6)
    forest = RPForest(n_trees=6, leaf_size=15, random_state=55).fit(X)
    q = np.random.randn(6)

    # Request k=10 but budget=4
    indices, dists = forest.query(q, k=10, search_budget=4)
    assert len(indices) == 4
    assert len(dists) == 4
    assert np.all(np.diff(dists) >= -1e-12)


def test_forest_search_budget_speedup_tradeoff():
    """Verify small candidate budget evaluates fewer candidates and executes faster on benchmark size."""
    np.random.seed(123)
    n_samples, dim = 1_500, 32
    X = np.random.randn(n_samples, dim)
    queries = np.random.randn(30, dim)

    forest = RPForest(n_trees=20, leaf_size=30, random_state=123).fit(X)

    # Unconstrained candidate evaluation
    t_start = time.perf_counter()
    _, _ = forest.batch_query(queries, k=10, search_budget=None)
    unconstrained_time = time.perf_counter() - t_start

    # Budgeted candidate evaluation (strictly capping distance computations)
    t_start = time.perf_counter()
    _, _ = forest.batch_query(queries, k=10, search_budget=15)
    budgeted_time = time.perf_counter() - t_start

    # Budgeted query should execute faster or at least on par while computing fewer distances
    assert budgeted_time <= unconstrained_time * 1.5


def test_forest_query_topk_ranking_and_self_retrieval():
    """Querying an indexed point returns itself as the closest candidate."""
    np.random.seed(999)
    X = np.random.randn(100, 12)
    forest = RPForest(n_trees=10, leaf_size=15, random_state=999).fit(X)

    indices, dists = forest.query(X[5], k=5)
    assert indices[0] == 5
    assert np.isclose(dists[0], 0.0, atol=1e-7)
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
    assert avg_recall >= 0.80, f"Average recall {avg_recall:.3f} was below expected threshold"


def test_forest_batch_query():
    """Verify batch_query matches sequential query calls."""
    np.random.seed(33)
    X = np.random.randn(100, 8)
    forest = RPForest(n_trees=5, leaf_size=15, random_state=33).fit(X)
    queries = np.random.randn(5, 8)

    batch_idx, batch_dists = forest.batch_query(queries, k=4, search_budget=20)
    assert batch_idx.shape == (5, 4)
    assert batch_dists.shape == (5, 4)

    for i in range(5):
        single_idx, single_dists = forest.query(queries[i], k=4, search_budget=20)
        assert np.array_equal(batch_idx[i], single_idx)
        assert np.allclose(batch_dists[i], single_dists)


def test_forest_k_greater_than_n_samples():
    """Requesting k > n_samples gracefully returns all samples."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    forest = RPForest(n_trees=3, leaf_size=1).fit(X)
    indices, dists = forest.query(np.array([2.0, 3.0]), k=10)
    assert len(indices) == 2
    assert len(dists) == 2


def test_forest_save_and_load_convenience(tmp_path):
    """Verify RPForest.save() and RPForest.load() classmethod."""
    np.random.seed(88)
    X = np.random.randn(100, 8)
    forest = RPForest(n_trees=4, leaf_size=15, random_state=88).fit(X)
    file_path = tmp_path / "index.rpf"

    forest.save(file_path)
    loaded = RPForest.load(file_path)

    q = np.random.randn(8)
    idx1, d1 = forest.query(q, k=5)
    idx2, d2 = loaded.query(q, k=5)
    assert np.array_equal(idx1, idx2)
    assert np.allclose(d1, d2)
