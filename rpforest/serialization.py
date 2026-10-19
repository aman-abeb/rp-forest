"""Binary serialization and deserialization for RPForest models."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from rpforest.tree import RPNode, RPTree

if TYPE_CHECKING:
    from rpforest.forest import RPForest

MAGIC_HEADER = b"RPFO"
FORMAT_VERSION = 1
HEADER_FIXED_SIZE = 56


def _count_nodes(node: RPNode | None) -> int:
    """Recursively count total nodes in a subtree."""
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


def _write_node(node: RPNode, buf: io.BytesIO) -> None:
    """Serialize a single tree node into the binary buffer in pre-order."""
    if node.is_leaf:
        buf.write(struct.pack("<?I", True, node.depth))
        indices = np.asarray(
            [] if node.indices is None else node.indices, dtype=np.int64
        )
        buf.write(struct.pack("<I", len(indices)))
        if len(indices) > 0:
            buf.write(indices.tobytes())
    else:
        buf.write(struct.pack("<?I", False, node.depth))
        split_v = float(node.split_val) if node.split_val is not None else 0.0
        buf.write(struct.pack("<d", split_v))
        if node.hyperplane is None:
            raise ValueError("Corrupted node: internal node missing hyperplane vector.")
        hp = np.asarray(node.hyperplane, dtype=np.float64)
        buf.write(hp.tobytes())
        if node.left is None or node.right is None:
            raise ValueError("Corrupted node: internal node missing child reference.")
        _write_node(node.left, buf)
        _write_node(node.right, buf)


def _read_node(stream: io.BytesIO, dim: int) -> RPNode:
    """Deserialize a single tree node from binary stream in pre-order."""
    node_hdr = stream.read(5)
    if len(node_hdr) < 5:
        raise ValueError("Corrupted index file: truncated node header.")
    is_leaf, depth = struct.unpack("<?I", node_hdr)

    if is_leaf:
        cnt_bytes = stream.read(4)
        if len(cnt_bytes) < 4:
            raise ValueError("Corrupted index file: truncated leaf indices count.")
        n_indices, = struct.unpack("<I", cnt_bytes)
        if n_indices > 0:
            ind_bytes = stream.read(n_indices * 8)
            if len(ind_bytes) < n_indices * 8:
                raise ValueError("Corrupted index file: truncated leaf indices array.")
            indices = np.frombuffer(ind_bytes, dtype=np.int64).copy()
        else:
            indices = np.array([], dtype=np.int64)
        return RPNode(is_leaf=True, indices=indices, depth=depth)
    else:
        split_bytes = stream.read(8)
        if len(split_bytes) < 8:
            raise ValueError("Corrupted index file: truncated split value.")
        split_val, = struct.unpack("<d", split_bytes)

        hp_bytes = stream.read(dim * 8)
        if len(hp_bytes) < dim * 8:
            raise ValueError("Corrupted index file: truncated hyperplane data.")
        hyperplane = np.frombuffer(hp_bytes, dtype=np.float64).copy()

        left = _read_node(stream, dim)
        right = _read_node(stream, dim)
        return RPNode(
            is_leaf=False,
            hyperplane=hyperplane,
            split_val=split_val,
            left=left,
            right=right,
            depth=depth,
        )


def save_forest(forest: RPForest, file_or_path: str | Path | BinaryIO) -> None:
    """Persist an RPForest index to disk or an open binary stream.

    The binary format consists of:
    1. 56-byte binary header with magic bytes, version, metadata length, payload length,
       and SHA-256 payload checksum for integrity validation.
    2. UTF-8 JSON metadata block.
    3. Raw vector data bytes (float64).
    4. Pre-order serialized decision nodes and leaf index arrays for each tree.

    Parameters
    ----------
    forest : RPForest
        Fitted forest instance to serialize.
    file_or_path : str | Path | BinaryIO
        Destination file path or writable binary stream.
    """
    if not forest.is_fitted or forest._data is None:
        raise RuntimeError("Cannot save unfitted RPForest index.")

    data = np.ascontiguousarray(forest._data, dtype=np.float64)
    n_samples, dimension = data.shape

    meta = {
        "metric": forest.metric,
        "n_trees": forest.n_trees,
        "leaf_size": forest.leaf_size,
        "max_depth": forest.max_depth,
        "n_samples": n_samples,
        "dimension": dimension,
    }
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")

    payload_buf = io.BytesIO()
    # 1. Vectors block: n_samples (uint64), dimension (uint32), dtype_code (uint8: 1=float64)
    payload_buf.write(struct.pack("<QIB", n_samples, dimension, 1))
    payload_buf.write(data.tobytes())

    # 2. Trees block: n_trees (uint32)
    payload_buf.write(struct.pack("<I", len(forest.trees)))
    for tree in forest.trees:
        max_d = -1 if tree.max_depth is None else tree.max_depth
        node_count = _count_nodes(tree.root)
        payload_buf.write(
            struct.pack(
                "<IiQII",
                tree.leaf_size,
                max_d,
                tree.n_samples,
                tree.dimension,
                node_count,
            )
        )
        if tree.root is not None:
            _write_node(tree.root, payload_buf)

    payload_bytes = payload_buf.getvalue()
    payload_len = len(payload_bytes)
    payload_sha = hashlib.sha256(payload_bytes).digest()

    # Header structure: <4s (magic), I (version), I (flags), I (meta_len), Q (payload_len), 32s (sha256)
    header = struct.pack(
        "<4sIIIQ32s",
        MAGIC_HEADER,
        FORMAT_VERSION,
        0,
        len(meta_bytes),
        payload_len,
        payload_sha,
    )

    serialized_bytes = header + meta_bytes + payload_bytes

    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "wb") as f:
            f.write(serialized_bytes)
    else:
        file_or_path.write(serialized_bytes)


def load_forest(file_or_path: str | Path | BinaryIO) -> RPForest:
    """Load an RPForest index from disk or an open binary stream.

    Validates magic bytes, format version, and SHA-256 payload checksum before
    reconstructing tree split nodes, leaf arrays, and vector data.

    Parameters
    ----------
    file_or_path : str | Path | BinaryIO
        Source file path or readable binary stream.

    Returns
    -------
    RPForest
        Reconstructed and fitted RPForest index.
    """
    # Import locally to avoid circular dependency
    from rpforest.forest import RPForest

    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            raw = f.read()
    elif hasattr(file_or_path, "read"):
        raw = file_or_path.read()
        if len(raw) == 0 and hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
            raw = file_or_path.read()
    else:
        raise TypeError(f"Unsupported file or path type: {type(file_or_path)}")

    if len(raw) < HEADER_FIXED_SIZE:
        raise ValueError(
            f"Invalid index file: size {len(raw)} bytes is smaller than header size {HEADER_FIXED_SIZE}."
        )

    magic, version, _flags, meta_len, payload_len, expected_sha = struct.unpack_from(
        "<4sIIIQ32s", raw, 0
    )

    if magic != MAGIC_HEADER:
        raise ValueError(
            f"Invalid file format: incorrect magic bytes (expected {MAGIC_HEADER!r}, got {magic!r})."
        )

    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version {version} (expected {FORMAT_VERSION})."
        )

    expected_total = HEADER_FIXED_SIZE + meta_len + payload_len
    if len(raw) < expected_total:
        raise ValueError(
            f"Corrupted index file: truncated file (expected {expected_total} bytes, got {len(raw)} bytes)."
        )

    meta_offset = HEADER_FIXED_SIZE
    meta_bytes = raw[meta_offset : meta_offset + meta_len]
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Corrupted metadata in index file: {exc}") from exc

    payload_offset = meta_offset + meta_len
    payload_bytes = raw[payload_offset : payload_offset + payload_len]

    computed_sha = hashlib.sha256(payload_bytes).digest()
    if computed_sha != expected_sha:
        raise ValueError(
            "Integrity check failed: index file payload is corrupted or modified (SHA-256 mismatch)."
        )

    stream = io.BytesIO(payload_bytes)

    # 1. Vectors
    n_samples, dimension, _dtype_code = struct.unpack("<QIB", stream.read(8 + 4 + 1))
    vec_bytes = stream.read(n_samples * dimension * 8)
    if len(vec_bytes) < n_samples * dimension * 8:
        raise ValueError("Corrupted index file: vector array truncated.")
    data = np.frombuffer(vec_bytes, dtype=np.float64).reshape((n_samples, dimension)).copy()

    # 2. Trees
    n_trees_bytes = stream.read(4)
    if len(n_trees_bytes) < 4:
        raise ValueError("Corrupted index file: truncated tree count.")
    n_trees, = struct.unpack("<I", n_trees_bytes)

    trees: list[RPTree] = []
    for _ in range(n_trees):
        tree_hdr = stream.read(4 + 4 + 8 + 4 + 4)
        if len(tree_hdr) < 24:
            raise ValueError("Corrupted index file: truncated tree header.")
        leaf_size, max_d, tree_n_samples, tree_dim, node_count = struct.unpack(
            "<IiQII", tree_hdr
        )
        max_depth = None if max_d < 0 else max_d

        tree = RPTree(leaf_size=leaf_size, max_depth=max_depth)
        tree._n_samples = tree_n_samples
        tree._dimension = tree_dim

        if node_count > 0:
            tree.root = _read_node(stream, tree_dim)
        trees.append(tree)

    forest = RPForest(
        n_trees=meta["n_trees"],
        leaf_size=meta["leaf_size"],
        max_depth=meta["max_depth"],
        metric=meta["metric"],
    )
    forest._data = data
    forest.trees = trees

    return forest
