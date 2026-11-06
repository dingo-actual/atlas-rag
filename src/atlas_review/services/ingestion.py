"""Explicit source ingestion: read, parse, preview, and store a corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from atlas_review.contracts import GenerationManifest, JsonValue
from atlas_review.corpus.atlas_v6 import PIPELINE_VERSION, ParsedCorpus, parse_source
from atlas_review.errors import (
    STAGE_SOURCE_ACQUISITION,
    STAGE_SOURCE_SCHEMA,
    AtlasReviewError,
    validation_details,
)
from atlas_review.hashing import fingerprint
from atlas_review.storage.repository import CorpusRepository


@dataclass(frozen=True, slots=True)
class SourcePreview:
    """Operator-facing summary of a parsed source before it is stored.

    Instance Attributes:
        origin (str): Path or URL the bytes came from.
        byte_count (int): Source size in bytes.
        source_sha256 (str): Digest of the exact bytes.
        format_version (str): Declared ATLAS format version.
        collection_version (str): Declared ATLAS collection version.
        entity_counts (tuple[tuple[str, int], ...]): Counts per record kind.
        edge_counts (tuple[tuple[str, int], ...]): Counts per relationship type.
        warnings (tuple[str, ...]): Sanitized parse observations.
    """

    origin: str
    byte_count: int
    source_sha256: str
    format_version: str
    collection_version: str
    entity_counts: tuple[tuple[str, int], ...]
    edge_counts: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    """Result of one explicit ingestion run.

    Instance Attributes:
        preview (SourcePreview): Summary of the parsed source.
        generation_id (str): Identifier of the stored generation.
    """

    preview: SourcePreview
    generation_id: str


def read_source_file(path: Path, *, max_bytes: int) -> bytes:
    """Read a local ATLAS source file within the configured size limit.

    Args:
        path (Path): Source file selected by the operator.
        max_bytes (int): Largest accepted source size.

    Returns:
        (bytes) Exact source bytes.

    Raises:
        AtlasReviewError: The file is missing, unreadable, empty, or too large.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _acquisition_error(
            "missing_source", f"cannot read source file {path}"
        ) from exc
    if size == 0:
        raise _acquisition_error("empty_source", f"source file {path} is empty")
    if size > max_bytes:
        raise _acquisition_error(
            "source_too_large",
            f"source file {path} exceeds the {max_bytes}-byte limit",
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _acquisition_error(
            "unreadable_source", f"cannot read source file {path}"
        ) from exc


def preview_source(raw: bytes, *, origin: str) -> tuple[ParsedCorpus, SourcePreview]:
    """Parse source bytes and summarize them without storing anything.

    Args:
        raw (bytes): Exact source bytes.
        origin (str): Path or URL for the snapshot.

    Returns:
        (tuple[ParsedCorpus, SourcePreview]) Parsed corpus and its summary.

    Raises:
        AtlasReviewError: The source is unsupported, internally inconsistent,
            or produces records that fail validation.
    """
    try:
        corpus = parse_source(
            raw,
            origin=origin,
            acquired_at=_timestamp(),
            license="Apache-2.0",
        )
    except ValidationError as exc:
        raise AtlasReviewError(
            code="invalid_source_payload",
            stage=STAGE_SOURCE_SCHEMA,
            message="the source produced records that failed validation",
            details=validation_details(exc),
            recovery="Correct the source records reported above and ingest again.",
        ) from exc
    return corpus, _preview(corpus, len(raw))


def ingest_source(
    repository: CorpusRepository, raw: bytes, *, origin: str
) -> IngestionOutcome:
    """Parse and store one source as a new generation.

    Increment 1 stores the normalized corpus only; index publication and
    generation activation arrive with the retrieval increment.

    Args:
        repository (CorpusRepository): Workspace repository.
        raw (bytes): Exact source bytes.
        origin (str): Path or URL for the snapshot.

    Returns:
        (IngestionOutcome) Stored generation identifier and preview.

    Raises:
        AtlasReviewError: The source is invalid or already stored.
    """
    corpus, preview = preview_source(raw, origin=origin)
    repository.store_generation(
        corpus,
        _manifest(corpus, created_at=_timestamp()),
        raw,
    )
    return IngestionOutcome(preview=preview, generation_id=corpus.generation_id)


def _manifest(corpus: ParsedCorpus, *, created_at: str) -> GenerationManifest:
    """Build the validated manifest recorded with a generation."""
    seed: JsonValue = {
        "source_sha256": corpus.snapshot.source_sha256,
        "pipeline": PIPELINE_VERSION,
        "format_version": corpus.snapshot.format_version,
    }
    return GenerationManifest(
        schema_version=1,
        generation_id=corpus.generation_id,
        created_at=created_at,
        pipeline_version=PIPELINE_VERSION,
        build_fingerprint=fingerprint(seed),
        snapshot=corpus.snapshot,
        entity_count=len(corpus.entities),
        edge_count=len(corpus.edges),
        warnings=list(corpus.warnings),
    )


def _preview(corpus: ParsedCorpus, byte_count: int) -> SourcePreview:
    """Summarize a parsed corpus for display."""
    entity_counts = Counter(entity.kind for entity in corpus.entities)
    edge_counts = Counter(edge.relation for edge in corpus.edges)
    return SourcePreview(
        origin=corpus.snapshot.origin,
        byte_count=byte_count,
        source_sha256=corpus.snapshot.source_sha256,
        format_version=corpus.snapshot.format_version,
        collection_version=corpus.snapshot.collection_version,
        entity_counts=tuple(sorted(entity_counts.items())),
        edge_counts=tuple(sorted(edge_counts.items())),
        warnings=corpus.warnings,
    )


def _timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()


def _acquisition_error(code: str, message: str) -> AtlasReviewError:
    """Build a source-acquisition error with a file recovery hint."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_SOURCE_ACQUISITION,
        message=message,
        recovery="Select a readable ATLAS v6 YAML file and try again.",
    )
