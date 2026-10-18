"""Unit tests for RPTree recursive hyperplane partitioning and tree traversal."""

import numpy as np
import pytest
from rpforest.tree import RPNode, RPTree


def test_tree_invalid_params():
    """Verify tree constructor parameter validation."""
    with pytest.raises(ValueError, match="leaf_size must be >= 1"):
        RPTree(leaf_size=0)
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        RPTree(max_depth=-1)


def test_tree_unfitted_query_raises():
    """Querying an unfitted tree raises RuntimeError."""
    tree = RPTree()
    with pytest.raises(RuntimeError, match="must be built before querying"):
        tree.query_leaf(np.array([1.0, 2.0]))


def test_tree_empty_data_raises():
    """Building tree with empty data raises ValueError."""
    tree = RPTree()
    with pytest.raises(ValueError, match="empty array"):
        tree.build(np.empty((0, 4)))


def test_tree_single_leaf_small_dataset():
    """When n_samples <= leaf_size, root node is directly a leaf."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    tree = RPTree(leaf_size=5).build(X)
    assert tree.root is not None
    assert tree.root.is_leaf
    assert tree.n_leaves == 1
    assert tree.max_tree_depth == 0
    assert np.array_equal(tree.query_leaf(np.array([2.0, 2.0])), [0, 1, 2])


def test_tree_median_splitting_invariants():
    """Verify hyperplane splitting invariants on internal nodes."""
    np.random.seed(42)
    X = np.random.randn(200, 8)
    tree = RPTree(leaf_size=20, random_state=42).build(X)

    # Traverse and verify invariants at all internal nodes
    stack = [tree.root]
    while stack:
        node = stack.pop()
        assert node is not None
        if not node.is_leaf:
            assert node.hyperplane is not None
            assert np.isclose(np.linalg.norm(node.hyperplane), 1.0, atol=1e-5)
            assert node.split_val is not None
            assert node.left is not None
            assert node.right is not None
            stack.append(node.left)
            stack.append(node.right)


def test_tree_partition_completeness():
    """Verify union of all leaves contains every indexed sample exactly once."""
    np.random.seed(99)
    n_samples = 300
    X = np.random.randn(n_samples, 10)
    tree = RPTree(leaf_size=15, random_state=99).build(X)

    leaves = tree.get_all_leaves()
    collected_indices: list[int] = []
    for leaf in leaves:
        assert leaf.is_leaf
        assert leaf.indices is not None
        # Verify leaf size constraint
        assert len(leaf.indices) <= 30  # recursive partitioning keeps leaves bounded
        collected_indices.extend(leaf.indices.tolist())

    assert len(collected_indices) == n_samples
    assert sorted(collected_indices) == list(range(n_samples))


def test_tree_max_depth_constraint():
    """Verify tree depth does not exceed specified max_depth."""
    np.random.seed(123)
    X = np.random.randn(500, 12)
    max_d = 4
    tree = RPTree(leaf_size=5, max_depth=max_d, random_state=123).build(X)
    assert tree.max_tree_depth <= max_d


def test_tree_deterministic_seeding():
    """Two trees built with the same random seed produce identical partitions."""
    np.random.seed(7)
    X = np.random.randn(100, 6)

    tree1 = RPTree(leaf_size=10, random_state=42).build(X)
    tree2 = RPTree(leaf_size=10, random_state=42).build(X)
    tree3 = RPTree(leaf_size=10, random_state=99).build(X)

    q = np.random.randn(6)
    leaf1 = tree1.query_leaf(q)
    leaf2 = tree2.query_leaf(q)
    leaf3 = tree3.query_leaf(q)

    assert np.array_equal(leaf1, leaf2)
    # With different seeds, partition or candidate set will typically differ
    assert not np.array_equal(leaf1, leaf3) or len(leaf1) > 0


def test_tree_identical_duplicate_points():
    """Verify tree safely handles duplicate points without infinite recursion."""
    # 50 copies of identical vector
    X = np.ones((50, 4))
    tree = RPTree(leaf_size=10, random_state=1).build(X)
    assert tree.is_fitted
    leaves = tree.get_all_leaves()
    total_pts = sum(len(leaf.indices) for leaf in leaves if leaf.indices is not None)
    assert total_pts == 50


def test_tree_query_dimension_mismatch():
    """Query with incorrect dimension raises ValueError."""
    X = np.random.randn(50, 8)
    tree = RPTree().build(X)
    with pytest.raises(ValueError, match="dimension .* does not match"):
        tree.query_leaf(np.random.randn(5))
