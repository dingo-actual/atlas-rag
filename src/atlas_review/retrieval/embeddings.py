"""FastEmbed embedding adapter plus vector validation and normalization.

The BGE query instruction is applied by this application exactly once, to the
query only. Passage text is embedded unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

from atlas_review.config import EmbeddingConfig
from atlas_review.errors import STAGE_EMBEDDING, AtlasReviewError

_MATRIX_DIMENSIONS = 2
_MIN_NORM = 1e-8


class EmbeddingBackend(Protocol):
    """Embed a batch of texts into one vector per text."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return unnormalized vectors in input order."""
        ...


class FastEmbedBackend:
    """Embed text with the fixed FastEmbed model on CPU.

    Instance Attributes:
        model (TextEmbedding): Lazily reusable embedding session.
    """

    def __init__(self, config: EmbeddingConfig, cache_dir: Path) -> None:
        """Load or download the configured embedding model.

        Args:
            config (EmbeddingConfig): Fixed model, batch, and thread settings.
            cache_dir (Path): Workspace model cache directory.
        """
        self._config = config
        self._model = TextEmbedding(
            model_name=config.model_name,
            cache_dir=str(cache_dir),
            threads=config.threads,
            providers=[config.provider],
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one raw vector per input text."""
        vectors = list(
            self._model.embed(list(texts), batch_size=self._config.batch_size)
        )
        return np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])


def passage_vectors(
    backend: EmbeddingBackend, config: EmbeddingConfig, texts: Sequence[str]
) -> np.ndarray:
    """Embed passage text without any query instruction.

    Args:
        backend (EmbeddingBackend): Embedding implementation.
        config (EmbeddingConfig): Declared dimension and instructions.
        texts (Sequence[str]): Augmented retrieval texts.

    Returns:
        (np.ndarray) Validated, L2-normalized contiguous float32 matrix.

    Raises:
        AtlasReviewError: The vector count, shape, or values are invalid.
    """
    if not texts:
        return np.zeros((0, config.dimension), dtype=np.float32)
    return _validated(backend.embed(texts), len(texts), config.dimension)


def query_vector(
    backend: EmbeddingBackend, config: EmbeddingConfig, focus: str
) -> np.ndarray:
    """Embed the retrieval focus with the BGE instruction applied once.

    Args:
        backend (EmbeddingBackend): Embedding implementation.
        config (EmbeddingConfig): Declared dimension and query instruction.
        focus (str): Ordinary retrieval focus text.

    Returns:
        (np.ndarray) Validated, normalized single query vector.

    Raises:
        AtlasReviewError: The returned vector is unusable.
    """
    vector: np.ndarray = _validated(
        backend.embed([config.query_prefix + focus]), 1, config.dimension
    )[0]
    return vector


def _validated(raw: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    """Validate shape, finiteness, and non-zero norms, then normalize."""
    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.ndim != _MATRIX_DIMENSIONS or vectors.shape != (rows, dimension):
        raise _embedding_error(
            f"expected a ({rows}, {dimension}) vector matrix, "
            f"received shape {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise _embedding_error("embedding vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < _MIN_NORM):
        raise _embedding_error("embedding vectors contain a near-zero norm")
    normalized: np.ndarray = np.ascontiguousarray(vectors / norms, dtype=np.float32)
    return normalized


def _embedding_error(message: str) -> AtlasReviewError:
    """Build an embedding-stage error."""
    return AtlasReviewError(
        code="invalid_embedding",
        stage=STAGE_EMBEDDING,
        message=message,
        recovery="Check the embedding model artifacts and configuration, then rebuild.",
    )
