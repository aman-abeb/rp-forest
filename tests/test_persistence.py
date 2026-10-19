"""Unit tests for binary persistence, integrity validation, and fidelity of RPForest."""

import io
from pathlib import Path
import numpy as np
import pytest

from rpforest.forest import RPForest
from rpforest.serialization import load_forest, save_forest


def test_persistence_roundtrip_file_euclidean(tmp_path: Path):
    """Verify file roundtrip serialization fidelity with Euclidean metric."""
    np.random.seed(42)
    X = np.random.randn(250, 16)
    queries = np.random.randn(10, 16)

    forest = RPForest(n_trees=6, leaf_size=20, metric="euclidean", random_state=42).fit(X)
    save_path = tmp_path / "forest_euc.rpf"
    forest.save(save_path)

    loaded = RPForest.load(save_path)
    assert loaded.is_fitted
    assert loaded.n_samples == forest.n_samples
    assert loaded.dimension == forest.dimension
    assert loaded.metric == "euclidean"
    assert loaded.n_trees == forest.n_trees
    assert loaded.leaf_size == forest.leaf_size
    assert loaded.max_depth == forest.max_depth
    assert np.array_equal(loaded._data, forest._data)

    # Test query fidelity across multiple test queries
    for q in queries:
        orig_idx, orig_dists = forest.query(q, k=7)
        load_idx, load_dists = loaded.query(q, k=7)
        assert np.array_equal(orig_idx, load_idx)
        assert np.allclose(orig_dists, load_dists)


def test_persistence_roundtrip_file_cosine(tmp_path: Path):
    """Verify file roundtrip serialization fidelity with Cosine metric."""
    np.random.seed(101)
    X = np.random.randn(150, 8)
    queries = np.random.randn(5, 8)

    forest = RPForest(n_trees=5, leaf_size=15, metric="cosine", random_state=101).fit(X)
    save_path = tmp_path / "forest_cos.rpf"
    save_forest(forest, save_path)

    loaded = load_forest(save_path)
    assert loaded.metric == "cosine"

    for q in queries:
        orig_idx, orig_dists = forest.query(q, k=5)
        load_idx, load_dists = loaded.query(q, k=5)
        assert np.array_equal(orig_idx, load_idx)
        assert np.allclose(orig_dists, load_dists)


def test_persistence_roundtrip_bytes_io():
    """Verify serialization and deserialization via in-memory BytesIO stream."""
    np.random.seed(7)
    X = np.random.randn(120, 10)
    forest = RPForest(n_trees=4, leaf_size=12, random_state=7).fit(X)

    buf = io.BytesIO()
    save_forest(forest, buf)
    buf.seek(0)

    loaded = load_forest(buf)
    assert loaded.is_fitted
    assert loaded.n_samples == 120

    q = np.random.randn(10)
    idx1, d1 = forest.query(q, k=4)
    idx2, d2 = loaded.query(q, k=4)
    assert np.array_equal(idx1, idx2)
    assert np.allclose(d1, d2)


def test_persistence_save_unfitted_raises():
    """Saving an unfitted index raises RuntimeError."""
    forest = RPForest()
    buf = io.BytesIO()
    with pytest.raises(RuntimeError, match="Cannot save unfitted"):
        save_forest(forest, buf)


def test_persistence_invalid_magic_bytes(tmp_path: Path):
    """Files with invalid magic bytes are rejected with ValueError."""
    np.random.seed(1)
    forest = RPForest(n_trees=2, leaf_size=10).fit(np.random.randn(30, 4))
    path = tmp_path / "corrupted_magic.rpf"
    forest.save(path)

    # Overwrite magic bytes
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(b"BAD!")

    with pytest.raises(ValueError, match="incorrect magic bytes"):
        RPForest.load(path)


def test_persistence_corrupted_payload_checksum(tmp_path: Path):
    """Payload bit flips trigger SHA-256 integrity validation failure."""
    np.random.seed(2)
    forest = RPForest(n_trees=3, leaf_size=10).fit(np.random.randn(50, 4))
    path = tmp_path / "corrupted_payload.rpf"
    forest.save(path)

    # Read file and flip byte in payload
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # Flip byte inside the payload
    data[-10] ^= 0xFF

    with open(path, "wb") as f:
        f.write(data)

    with pytest.raises(ValueError, match="Integrity check failed"):
        RPForest.load(path)


def test_persistence_truncated_file(tmp_path: Path):
    """Truncated files raise ValueError."""
    np.random.seed(3)
    forest = RPForest(n_trees=3, leaf_size=10).fit(np.random.randn(40, 4))
    path = tmp_path / "truncated.rpf"
    forest.save(path)

    with open(path, "rb") as f:
        data = f.read()

    # Truncate halfway
    truncated_data = data[: len(data) // 2]
    with open(path, "wb") as f:
        f.write(truncated_data)

    with pytest.raises(ValueError, match="truncated|smaller"):
        RPForest.load(path)


def test_persistence_unsupported_version(tmp_path: Path):
    """Unsupported format version raises ValueError."""
    np.random.seed(4)
    forest = RPForest(n_trees=2, leaf_size=10).fit(np.random.randn(30, 4))
    path = tmp_path / "bad_version.rpf"
    forest.save(path)

    with open(path, "r+b") as f:
        # Version is uint32 at offset 4
        f.seek(4)
        f.write((999).to_bytes(4, byteorder="little"))

    with pytest.raises(ValueError, match="Unsupported format version"):
        RPForest.load(path)


def test_persistence_with_search_budget(tmp_path: Path):
    """Verify query fidelity persists when using search budget on loaded forest."""
    np.random.seed(42)
    X = np.random.randn(200, 8)
    queries = np.random.randn(5, 8)
    forest = RPForest(n_trees=8, leaf_size=15, random_state=42).fit(X)
    path = tmp_path / "budgeted.rpf"
    forest.save(path)

    loaded = RPForest.load(path)

    orig_batch_idx, orig_batch_dist = forest.batch_query(queries, k=5, search_budget=20)
    load_batch_idx, load_batch_dist = loaded.batch_query(queries, k=5, search_budget=20)

    assert np.array_equal(orig_batch_idx, load_batch_idx)
    assert np.allclose(orig_batch_dist, load_batch_dist)
