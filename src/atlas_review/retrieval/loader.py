"""Load one ready generation's bundle and verify joint index integrity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import bm25s
from usearch.index import Index

from atlas_review.config import AppConfig
from atlas_review.contracts import Passage
from atlas_review.errors import STAGE_INDEX, STAGE_RETRIEVAL, AtlasReviewError
from atlas_review.retrieval.artifacts import (
    MANIFEST_NAME,
    decode_passage_ids,
    load_artifact_manifest,
)
from atlas_review.retrieval.keyword_index import load_keyword_index
from atlas_review.retrieval.vector_index import restore_dense_index
from atlas_review.storage.passages import PassageRepository
from atlas_review.storage.repository import CorpusRepository


@dataclass(frozen=True, slots=True)
class LoadedGeneration:
    """A ready generation with both indexes, the mapping, and its passages.

    Instance Attributes:
        generation_id (str): Active generation identifier.
        collection_version (str): ATLAS collection version of the generation.
        passage_ids (tuple[int, ...]): Passage IDs in BM25 ordinal order.
        passages (dict[int, Passage]): Passage records by identifier.
        dense_index (Index): Restored USearch index.
        keyword_index (bm25s.BM25): Restored bm25s index.
    """

    generation_id: str
    collection_version: str
    passage_ids: tuple[int, ...]
    passages: dict[int, Passage]
    dense_index: Index
    keyword_index: bm25s.BM25


def load_generation(
    repository: CorpusRepository,
    *,
    passages: PassageRepository,
    indexes_dir: Path,
    config: AppConfig,
    generation_id: str,
) -> LoadedGeneration:
    """Load and verify one generation's dense and keyword indexes together.

    Args:
        repository (CorpusRepository): Workspace repository.
        passages (PassageRepository): Passage storage.
        indexes_dir (Path): Workspace index directory.
        config (AppConfig): Query-side configuration to compare with the bundle.
        generation_id (str): Generation to load.

    Returns:
        (LoadedGeneration) Verified bundle, mapping, and passages.

    Raises:
        AtlasReviewError: An artifact, hash, key set, or setting is incompatible.
    """
    artifact = repository.artifact(generation_id)
    if artifact is None:
        raise _selector_error(
            "missing_index_bundle",
            f"generation {generation_id} has no published index bundle",
        )
    directory = _bundle_path(indexes_dir, artifact.directory)
    manifest_bytes = _read_manifest_bytes(directory, generation_id)
    if hashlib.sha256(manifest_bytes).hexdigest() != artifact.manifest_sha256:
        raise _selector_error(
            "artifact_manifest_mismatch",
            f"the manifest of generation {generation_id} was modified",
        )
    manifest = load_artifact_manifest(directory)
    _verify_files(directory, manifest)
    _verify_manifest(manifest, config, generation_id)
    passage_ids = decode_passage_ids(directory)
    stored = passages.passages(generation_id)
    by_id = {passage.passage_id: passage for passage in stored}
    if passage_ids != tuple(sorted(by_id, key=lambda key: by_id[key].bm25_ordinal)):
        raise _selector_error(
            "keyword_mapping_mismatch",
            "the stored passage mapping does not match the stored passages",
        )
    if list(passage_ids) != manifest.get("passage_ids"):
        raise _selector_error(
            "keyword_mapping_mismatch",
            "the stored passage mapping does not match the bundle manifest",
        )
    dense = restore_dense_index(directory / "dense.usearch")
    if dense.ndim != config.embedding.dimension or len(dense) != len(passage_ids):
        raise _selector_error(
            "dense_index_mismatch", "the dense index does not match the generation"
        )
    if sorted(int(key) for key in dense.keys) != sorted(passage_ids):
        raise _selector_error(
            "dense_index_mismatch", "the dense keys do not match the passage IDs"
        )
    keyword = load_keyword_index(directory / "keyword")
    if int(keyword.scores["num_docs"]) != len(passage_ids):
        raise _selector_error(
            "keyword_index_mismatch",
            "the keyword index does not match the passage population",
        )
    return LoadedGeneration(
        generation_id=generation_id,
        collection_version=repository.snapshot(generation_id).collection_version,
        passage_ids=passage_ids,
        passages=by_id,
        dense_index=dense,
        keyword_index=keyword,
    )


def _read_manifest_bytes(directory: Path, generation_id: str) -> bytes:
    """Read the published bundle manifest bytes or report a missing bundle."""
    try:
        return (directory / MANIFEST_NAME).read_bytes()
    except OSError as exc:
        raise _selector_error(
            "missing_artifact_manifest",
            f"generation {generation_id} has no published bundle manifest",
        ) from exc


def _bundle_path(indexes_dir: Path, recorded: str) -> Path:
    """Resolve a recorded bundle path without escaping the index directory."""
    directory = (indexes_dir.parent / recorded).resolve()
    if not directory.is_relative_to(indexes_dir.resolve()) or not directory.is_dir():
        raise _selector_error(
            "missing_index_bundle", "the recorded bundle path is not usable"
        )
    return directory


def _verify_files(directory: Path, manifest: dict[str, object]) -> None:
    """Check every listed artifact still matches its recorded hash."""
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise _selector_error("artifact_manifest_mismatch", "no artifacts are listed")
    for entry in entries:
        if not isinstance(entry, dict):
            raise _selector_error(
                "artifact_manifest_mismatch", "malformed artifact entry"
            )
        name = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if not isinstance(name, str) or directory.joinpath(name).is_symlink():
            raise _selector_error(
                "artifact_manifest_mismatch", "an artifact path is not usable"
            )
        path = directory / name
        if not path.is_relative_to(directory) or not path.is_file():
            raise _selector_error(
                "missing_index_artifact", f"bundle artifact {name} is missing"
            )
        payload = path.read_bytes()
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise _selector_error(
                "artifact_hash_mismatch", f"bundle artifact {name} was modified"
            )


def _verify_manifest(
    manifest: dict[str, object], config: AppConfig, generation_id: str
) -> None:
    """Check the bundle belongs to this generation and these query settings."""
    if manifest.get("generation_id") != generation_id:
        raise _selector_error(
            "artifact_manifest_mismatch",
            "the bundle belongs to a different generation",
        )
    embedding = manifest.get("embedding")
    keyword = manifest.get("keyword")
    if not isinstance(embedding, dict) or not isinstance(keyword, dict):
        raise _selector_error("artifact_manifest_mismatch", "malformed settings")
    if (
        embedding.get("model_name") != config.embedding.model_name
        or embedding.get("query_prefix") != config.embedding.query_prefix
        or embedding.get("metric") != config.vector_index.metric
    ):
        raise _selector_error(
            "query_configuration_mismatch",
            "the embedding settings do not match the active generation",
        )
    if (
        keyword.get("analyzer_version") != config.keyword_index.analyzer_version
        or keyword.get("token_pattern") != config.keyword_index.token_pattern
        or keyword.get("method") != config.keyword_index.method
        or keyword.get("k1") != config.keyword_index.k1
        or keyword.get("b") != config.keyword_index.b
    ):
        raise _selector_error(
            "query_configuration_mismatch",
            "the keyword settings do not match the active generation",
        )


def _selector_error(code: str, message: str) -> AtlasReviewError:
    """Build an index or query-configuration error."""
    stage = STAGE_INDEX if code != "query_configuration_mismatch" else STAGE_RETRIEVAL
    return AtlasReviewError(
        code=code,
        stage=stage,
        message=message,
        recovery=(
            "Restore the configured query settings, or rebuild the index bundle "
            "explicitly from stored vectors and text."
        ),
    )
