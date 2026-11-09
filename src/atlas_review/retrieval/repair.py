"""Explicit index repair: rebuild both derived indexes from stored rows.

Repair uses only SQLite passages, stored vectors, and stored augmented text. It
never calls the embedding model, the contextualization provider, or any other
model. The generation ID and passage IDs stay unchanged, so saved reviews keep
referring to the same evidence.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from atlas_review.config import AppConfig
from atlas_review.contracts import GenerationManifest
from atlas_review.corpus.identity import with_artifact
from atlas_review.errors import (
    STAGE_INDEX,
    STAGE_RETRIEVAL,
    AtlasReviewError,
)
from atlas_review.retrieval.artifacts import Publication, build_bundle
from atlas_review.storage.passages import PassageRepository, StoredIndexRow
from atlas_review.storage.repository import CorpusRepository, StoredArtifact

ProgressCallback = Callable[[str], None]
_NORMALIZED_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class RepairServices:
    """Collaborators an explicit index repair needs.

    Instance Attributes:
        config (AppConfig): Current query-side configuration.
        repository (CorpusRepository): Generation and artifact storage.
        passages (PassageRepository): Stored passages, vectors, and text.
        indexes_dir (Path): Workspace index directory.
        progress (ProgressCallback | None): Stage progress reporter.
    """

    config: AppConfig
    repository: CorpusRepository
    passages: PassageRepository
    indexes_dir: Path
    progress: ProgressCallback | None


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """Result of one explicit index repair.

    Instance Attributes:
        generation_id (str): Generation whose bundle was rebuilt.
        passage_count (int): Passages covered by both indexes.
        dense_count (int): Vectors in the rebuilt dense index.
        keyword_count (int): Documents in the rebuilt keyword index.
        artifact_directory (str): Published bundle path inside the workspace.
    """

    generation_id: str
    passage_count: int
    dense_count: int
    keyword_count: int
    artifact_directory: str


def rebuild_indexes(generation_id: str, *, services: RepairServices) -> RepairOutcome:
    """Rebuild a ready generation's dense and keyword indexes from SQLite.

    Args:
        generation_id (str): Ready generation to repair.
        services (RepairServices): Collaborators for the repair.

    Returns:
        (RepairOutcome) Rebuilt bundle summary.

    Raises:
        AtlasReviewError: The generation is not repairable, the stored inputs
            disagree with the recorded settings, or publication fails. The
            previously recorded artifact paths stay unchanged on failure.
    """
    record = services.repository.generation(generation_id)
    if record is None or record.status != "ready":
        raise _repair_error(
            "generation_not_repairable",
            f"generation {generation_id} has no ready stored passages to rebuild",
        )
    manifest = services.repository.manifest(generation_id)
    previous = services.repository.artifact(generation_id)
    rows = services.passages.index_rows(generation_id)
    _check_inputs(manifest, rows, services.config)
    vectors = _vector_matrix(rows, services.config.embedding.dimension)
    report = services.progress if services.progress is not None else _ignore
    report(f"rebuilding both indexes from {len(rows)} stored passages")
    bundle = build_bundle(
        publication=Publication(
            indexes_dir=services.indexes_dir,
            generation_id=generation_id,
            artifact_id=uuid4().hex[:16],
        ),
        vectors=vectors,
        passage_ids=[row.passage_id for row in rows],
        retrieval_texts=[row.retrieval_text for row in rows],
        config=services.config,
    )
    artifact = StoredArtifact(
        directory=bundle.directory.relative_to(services.indexes_dir.parent).as_posix(),
        manifest_sha256=bundle.manifest_sha256,
    )
    with services.repository.transaction() as connection:
        services.repository.update_artifact(
            connection, manifest=with_artifact(manifest, artifact), artifact=artifact
        )
    _remove_superseded(previous, artifact, services, report)
    report(f"repaired generation {generation_id}")
    return RepairOutcome(
        generation_id=generation_id,
        passage_count=len(rows),
        dense_count=len(rows),
        keyword_count=len(rows),
        artifact_directory=artifact.directory,
    )


def _check_inputs(
    manifest: GenerationManifest,
    rows: Sequence[StoredIndexRow],
    config: AppConfig,
) -> None:
    """Check that stored rows are complete and match the recorded settings."""
    if _recorded_settings(manifest) != _expected_settings(config):
        raise AtlasReviewError(
            code="query_configuration_mismatch",
            stage=STAGE_RETRIEVAL,
            message="the current analysis settings do not match this generation",
            recovery=(
                "Restore the recorded settings, or re-ingest the source with the "
                "new settings instead of repairing."
            ),
        )
    if not rows:
        raise _repair_error(
            "empty_generation",
            f"generation {manifest.generation_id} stores no passages",
        )
    if [row.bm25_ordinal for row in rows] != list(range(len(rows))):
        raise _repair_error(
            "index_ordinal_mismatch",
            f"generation {manifest.generation_id} has non-contiguous BM25 ordinals",
        )
    if manifest.passage_count != len(rows):
        raise _repair_error(
            "passage_population_mismatch",
            f"generation {manifest.generation_id} records "
            f"{manifest.passage_count} passages but stores {len(rows)}",
        )


def _remove_superseded(
    previous: StoredArtifact | None,
    published: StoredArtifact,
    services: RepairServices,
    report: ProgressCallback,
) -> None:
    """Delete the replaced bundle once no stored generation references it."""
    if previous is None or previous.directory == published.directory:
        return
    if services.repository.artifact_is_referenced(previous.directory):
        return
    root = services.indexes_dir.resolve()
    directory = (services.indexes_dir.parent / previous.directory).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        return
    try:
        shutil.rmtree(directory)
    except OSError:
        report(f"could not remove superseded bundle {previous.directory}")


def _recorded_settings(manifest: GenerationManifest) -> dict[str, object]:
    """Return the analysis settings recorded with a generation."""
    embedding = manifest.embedding
    lexical = manifest.lexical
    if embedding is None or lexical is None:
        raise _repair_error(
            "generation_settings_missing",
            f"generation {manifest.generation_id} does not record its settings",
        )
    return {
        "embedding": {
            "model_name": embedding.model_name,
            "dimension": embedding.dimension,
            "query_prefix": embedding.query_prefix,
        },
        "lexical": {
            "analyzer_version": lexical.analyzer_version,
            "token_pattern": lexical.token_pattern,
            "lowercase": lexical.lowercase,
            "stopwords": list(lexical.stopwords),
            "stemming": lexical.stemming,
            "method": lexical.method,
            "k1": lexical.k1,
            "b": lexical.b,
            "dtype": lexical.dtype,
        },
    }


def _expected_settings(config: AppConfig) -> dict[str, object]:
    """Return the analysis settings the current configuration would use."""
    return {
        "embedding": {
            "model_name": config.embedding.model_name,
            "dimension": config.embedding.dimension,
            "query_prefix": config.embedding.query_prefix,
        },
        "lexical": {
            "analyzer_version": config.keyword_index.analyzer_version,
            "token_pattern": config.keyword_index.token_pattern,
            "lowercase": config.keyword_index.lowercase,
            "stopwords": list(config.keyword_index.stopwords),
            "stemming": config.keyword_index.stemming,
            "method": config.keyword_index.method,
            "k1": config.keyword_index.k1,
            "b": config.keyword_index.b,
            "dtype": config.keyword_index.dtype,
        },
    }


def _vector_matrix(rows: Sequence[StoredIndexRow], dimension: int) -> np.ndarray:
    """Decode and validate the stored float32 vectors in passage order."""
    expected_bytes = dimension * 4
    for row in rows:
        if len(row.vector) != expected_bytes:
            raise _repair_error(
                "invalid_stored_vector",
                f"passage {row.passage_id} does not hold a {dimension}-dimension "
                "vector",
            )
    matrix = np.vstack([np.frombuffer(row.vector, dtype="<f4") for row in rows]).astype(
        np.float32
    )
    if not np.isfinite(matrix).all():
        raise _repair_error(
            "invalid_stored_vector", "stored vectors contain non-finite values"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORMALIZED_TOLERANCE):
        raise _repair_error(
            "invalid_stored_vector", "stored vectors are not L2-normalized"
        )
    return np.ascontiguousarray(matrix)


def _ignore(message: str) -> None:
    """Discard a progress message when no reporter is supplied."""
    del message


def _repair_error(code: str, message: str) -> AtlasReviewError:
    """Build an index-repair error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_INDEX,
        message=message,
        recovery=(
            "Repair needs the stored passages and vectors of a ready generation; "
            "re-ingest the source instead."
        ),
    )
