"""Immutable index-bundle publication: dense, keyword, mapping, and manifest.

Both derived indexes are written into one temporary directory, validated, and
renamed into place as a single bundle. A bundle is never partially published.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import Field, TypeAdapter, ValidationError

from atlas_review.config import AppConfig
from atlas_review.errors import STAGE_INDEX, AtlasReviewError
from atlas_review.hashing import canonical_json
from atlas_review.retrieval.keyword_index import (
    build_keyword_index,
    load_keyword_index,
    save_keyword_index,
)
from atlas_review.retrieval.vector_index import (
    build_dense_index,
    restore_dense_index,
    save_dense_index,
)
from atlas_review.storage.codec import decode_plain, encode_plain

DENSE_NAME = "dense.usearch"
KEYWORD_DIRECTORY = "keyword"
MAPPING_NAME = "passage_ids.msgpack"
MANIFEST_NAME = "artifacts.json"
BUNDLE_SCHEMA_VERSION = 1

PassageIdList = list[Annotated[int, Field(strict=True, gt=0, lt=2**63)]]
_PASSAGE_IDS: TypeAdapter[PassageIdList] = TypeAdapter(PassageIdList)


@dataclass(frozen=True, slots=True)
class Publication:
    """Where one index bundle is published.

    Instance Attributes:
        indexes_dir (Path): Workspace index directory.
        generation_id (str): Generation the bundle belongs to.
        artifact_id (str): Unique bundle suffix for this publication.
    """

    indexes_dir: Path
    generation_id: str
    artifact_id: str


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """One published index bundle.

    Instance Attributes:
        directory (Path): Final bundle directory inside the workspace.
        manifest_sha256 (str): Digest of the stored ``artifacts.json`` bytes.
        files (tuple[tuple[str, int, str], ...]): Relative path, size, digest.
    """

    directory: Path
    manifest_sha256: str
    files: tuple[tuple[str, int, str], ...]


def build_bundle(
    *,
    publication: Publication,
    vectors: np.ndarray,
    passage_ids: Sequence[int],
    retrieval_texts: Sequence[str],
    config: AppConfig,
) -> ArtifactBundle:
    """Build, validate, and publish one immutable index bundle.

    Args:
        publication (Publication): Target directory and bundle identity.
        vectors (np.ndarray): Normalized vectors in passage order.
        passage_ids (Sequence[int]): Passage IDs in BM25-ordinal order.
        retrieval_texts (Sequence[str]): Augmented texts in the same order.
        config (AppConfig): Index, analyzer, and embedding settings.

    Returns:
        (ArtifactBundle) Published bundle with file hashes.

    Raises:
        AtlasReviewError: Inputs disagree or a validated index cannot be reloaded.
    """
    indexes_dir = publication.indexes_dir
    ids = _validated_ids(passage_ids, len(retrieval_texts), len(vectors))
    indexes_dir.mkdir(parents=True, exist_ok=True)
    final = indexes_dir / f"{publication.generation_id}-{publication.artifact_id}"
    temporary = indexes_dir / f".{final.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    index = build_dense_index(vectors, ids, config.vector_index)
    save_dense_index(index, temporary / DENSE_NAME)
    save_keyword_index(
        build_keyword_index(retrieval_texts, config.keyword_index),
        temporary / KEYWORD_DIRECTORY,
        config.keyword_index,
    )
    (temporary / MAPPING_NAME).write_bytes(
        encode_plain(_PASSAGE_IDS.dump_python(list(ids), mode="json"))
    )
    _verify_bundle_files(temporary, ids, config)
    files = _file_entries(temporary)
    manifest = _manifest(
        generation_id=publication.generation_id,
        artifact_id=publication.artifact_id,
        passage_ids=ids,
        files=files,
        config=config,
        vectors=vectors,
    )
    manifest_bytes = canonical_json(manifest).encode("utf-8")
    (temporary / MANIFEST_NAME).write_bytes(manifest_bytes)
    if final.exists():
        shutil.rmtree(final)
    temporary.rename(final)
    return ArtifactBundle(
        directory=final,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        files=files,
    )


def load_artifact_manifest(directory: Path) -> dict[str, Any]:
    """Read a bundle manifest.

    Args:
        directory (Path): Bundle directory.

    Returns:
        (dict[str, Any]) Parsed manifest mapping.

    Raises:
        AtlasReviewError: The manifest is missing or not a JSON object.
    """
    path = directory / MANIFEST_NAME
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise _bundle_error(
            "missing_artifact_manifest", f"cannot read {path.name} in {directory.name}"
        ) from exc
    if not isinstance(value, dict):
        raise _bundle_error(
            "invalid_artifact_manifest", f"{path.name} is not a JSON object"
        )
    return value


def decode_passage_ids(directory: Path) -> tuple[int, ...]:
    """Decode and validate the stored ordinal-to-passage-ID mapping.

    Args:
        directory (Path): Bundle directory.

    Returns:
        (tuple[int, ...]) Passage IDs in BM25 ordinal order.

    Raises:
        AtlasReviewError: The mapping is missing or fails validation.
    """
    try:
        payload = (directory / MAPPING_NAME).read_bytes()
    except OSError as exc:
        raise _bundle_error(
            "invalid_keyword_mapping", "the stored passage-ID mapping is missing"
        ) from exc
    return decode_passage_ids_payload(payload)


def decode_passage_ids_payload(payload: bytes) -> tuple[int, ...]:
    """Decode and validate an encoded ordinal-to-passage-ID mapping.

    Args:
        payload (bytes): MessagePack bytes of the mapping.

    Returns:
        (tuple[int, ...]) Passage IDs in BM25 ordinal order.

    Raises:
        AtlasReviewError: The payload is not a valid unique positive-ID list.
    """
    try:
        plain = decode_plain(payload)
        values = _PASSAGE_IDS.validate_python(plain, strict=True)
    except (AtlasReviewError, ValidationError) as exc:
        raise _bundle_error(
            "invalid_keyword_mapping", "the stored passage-ID mapping is invalid"
        ) from exc
    return tuple(values)


def encode_passage_ids(passage_ids: Sequence[int]) -> bytes:
    """Encode an ordered passage-ID mapping for storage.

    Args:
        passage_ids (Sequence[int]): Passage IDs in ordinal order.

    Returns:
        (bytes) MessagePack bytes of the validated mapping.
    """
    validated = _PASSAGE_IDS.validate_python(list(passage_ids), strict=True)
    return encode_plain(_PASSAGE_IDS.dump_python(validated, mode="json"))


def _validated_ids(
    passage_ids: Sequence[int], text_count: int, vector_count: int
) -> tuple[int, ...]:
    """Check that the three passage populations agree."""
    try:
        ids = tuple(_PASSAGE_IDS.validate_python(list(passage_ids), strict=True))
    except ValidationError as exc:
        raise _bundle_error(
            "invalid_passage_id", "a passage identifier is not a positive integer"
        ) from exc
    if len(ids) != text_count or len(ids) != vector_count:
        raise _bundle_error(
            "invalid_index_inputs",
            "the passage, text, and vector populations disagree",
        )
    if len(set(ids)) != len(ids):
        raise _bundle_error("invalid_passage_id", "the passage identifiers repeat")
    return ids


def _verify_bundle_files(
    directory: Path, ids: tuple[int, ...], config: AppConfig
) -> None:
    """Reload both indexes and check they cover the same passage set."""
    dense = restore_dense_index(directory / DENSE_NAME)
    if len(dense) != len(ids) or dense.ndim != config.embedding.dimension:
        raise _bundle_error(
            "index_population_mismatch", "the dense index does not match its inputs"
        )
    if sorted(int(key) for key in dense.keys) != list(ids):
        raise _bundle_error(
            "index_population_mismatch", "the dense keys do not match the passage IDs"
        )
    keyword = load_keyword_index(directory / KEYWORD_DIRECTORY)
    if int(keyword.scores["num_docs"]) != len(ids):
        raise _bundle_error(
            "index_population_mismatch",
            "the keyword index does not match the passage population",
        )
    if decode_passage_ids(directory) != ids:
        raise _bundle_error(
            "invalid_keyword_mapping", "the stored mapping does not match the inputs"
        )


def _file_entries(directory: Path) -> tuple[tuple[str, int, str], ...]:
    """Return hashed entries for every generated bundle file."""
    entries: list[tuple[str, int, str]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        payload = path.read_bytes()
        entries.append(
            (
                path.relative_to(directory).as_posix(),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(entries)


def _manifest(
    *,
    generation_id: str,
    artifact_id: str,
    passage_ids: tuple[int, ...],
    files: tuple[tuple[str, int, str], ...],
    config: AppConfig,
    vectors: np.ndarray,
) -> dict[str, Any]:
    """Build the bundle manifest mapping."""
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generation_id": generation_id,
        "artifact_id": artifact_id,
        "passage_count": len(passage_ids),
        "dimension": int(vectors.shape[1]),
        "passage_ids": list(passage_ids),
        "embedding": {
            "model_name": config.embedding.model_name,
            "query_prefix": config.embedding.query_prefix,
            "metric": config.vector_index.metric,
            "dtype": config.vector_index.dtype,
        },
        "keyword": {
            "analyzer_version": config.keyword_index.analyzer_version,
            "token_pattern": config.keyword_index.token_pattern,
            "method": config.keyword_index.method,
            "k1": config.keyword_index.k1,
            "b": config.keyword_index.b,
        },
        "files": [
            {"path": name, "bytes": size, "sha256": digest}
            for name, size, digest in files
        ],
    }


def _bundle_error(code: str, message: str) -> AtlasReviewError:
    """Build an index-bundle error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_INDEX,
        message=message,
        recovery="Rebuild the index bundle explicitly from stored vectors and text.",
    )
