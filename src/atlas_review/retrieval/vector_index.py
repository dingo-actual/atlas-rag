"""USearch dense index construction, persistence, and search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from usearch.index import Index, Matches

from atlas_review.config import VectorIndexConfig
from atlas_review.errors import STAGE_INDEX, AtlasReviewError
from atlas_review.hashing import sha256_hex

_MATRIX_DIMENSIONS = 2


def checked_passage_ids(passage_ids: Sequence[int]) -> np.ndarray:
    """Convert positive SQLite passage IDs to USearch keys.

    Args:
        passage_ids (Sequence[int]): Positive signed 64-bit passage IDs.

    Returns:
        (np.ndarray) ``uint64`` keys in input order.

    Raises:
        AtlasReviewError: An identifier is not a positive signed 64-bit integer.
    """
    for passage_id in passage_ids:
        if type(passage_id) is not int or not 0 < passage_id < 2**63:
            raise _index_error(
                "invalid_passage_id",
                f"passage identifier {passage_id!r} is not a positive signed integer",
            )
    return np.asarray(passage_ids, dtype=np.uint64)


def build_dense_index(
    vectors: np.ndarray, passage_ids: Sequence[int], config: VectorIndexConfig
) -> Index:
    """Build a USearch index keyed by SQLite passage IDs.

    Args:
        vectors (np.ndarray): Normalized float32 vectors in passage order.
        passage_ids (Sequence[int]): Passage IDs matching the vector rows.
        config (VectorIndexConfig): Fixed index construction parameters.

    Returns:
        (Index) Populated dense index.

    Raises:
        AtlasReviewError: Rows and identifiers disagree or construction failed.
    """
    keys = checked_passage_ids(passage_ids)
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != _MATRIX_DIMENSIONS or matrix.shape[0] != len(keys):
        raise _index_error(
            "invalid_index_inputs",
            "vector rows do not match the passage identifier count",
        )
    index = Index(
        ndim=matrix.shape[1],
        metric=config.metric,
        dtype=config.dtype,
        connectivity=config.connectivity,
        expansion_add=config.expansion_add,
        expansion_search=config.expansion_search,
        multi=False,
    )
    if len(keys):
        index.add(keys, matrix, threads=config.threads)
    return index


def save_dense_index(index: Index, path: Path) -> None:
    """Persist a dense index to one file."""
    index.save(str(path))


def restore_dense_index(path: Path) -> Index:
    """Restore a dense index from one file.

    Args:
        path (Path): Saved USearch file.

    Returns:
        (Index) Restored index.

    Raises:
        AtlasReviewError: The file is missing or unreadable.
    """
    try:
        restored = Index.restore(str(path), view=False)
    except (OSError, RuntimeError) as exc:
        raise _index_error(
            "unreadable_dense_index", f"cannot restore {path.name}"
        ) from exc
    if restored is None:
        raise _index_error("unreadable_dense_index", f"cannot restore {path.name}")
    return restored


def dense_search(
    index: Index, query: np.ndarray, *, count: int, threads: int
) -> tuple[tuple[int, float], ...]:
    """Search the dense index and return ``(passage_id, distance)`` pairs.

    Args:
        index (Index): Loaded dense index.
        query (np.ndarray): Normalized one-dimensional query vector.
        count (int): Maximum matches, bounded by the index size.
        threads (int): Search threads.

    Returns:
        (tuple[tuple[int, float], ...]) Hits ordered by ascending distance.

    Raises:
        AtlasReviewError: The index returns keys outside the signed 64-bit range.
    """
    limit = min(count, len(index))
    if limit <= 0:
        return ()
    matches = index.search(
        np.ascontiguousarray(query, dtype=np.float32),
        count=limit,
        exact=False,
        threads=threads,
    )
    if not isinstance(matches, Matches):
        raise _index_error(
            "dense_search_failed", "dense search returned a batch result"
        )
    hits: list[tuple[int, float]] = []
    for key, distance in zip(matches.keys, matches.distances, strict=True):
        passage_id = int(key)
        if not 0 < passage_id < 2**63:
            raise _index_error(
                "invalid_dense_key", "dense index returned an unusable passage key"
            )
        hits.append((passage_id, float(distance)))
    return tuple(sorted(hits, key=lambda hit: (hit[1], hit[0])))


def dense_index_digest(path: Path) -> str:
    """Return the digest of a saved dense index file."""
    return sha256_hex(path.read_bytes())


def _index_error(code: str, message: str) -> AtlasReviewError:
    """Build an index-integrity error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_INDEX,
        message=message,
        recovery="Rebuild the index bundle explicitly from stored vectors and text.",
    )
