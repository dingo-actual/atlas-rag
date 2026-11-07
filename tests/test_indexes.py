"""Embedding, USearch, bm25s, and index-bundle behaviour."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from atlas_review.config import AppConfig
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval import embeddings
from atlas_review.retrieval.artifacts import (
    ArtifactBundle,
    Publication,
    build_bundle,
    load_artifact_manifest,
)
from atlas_review.retrieval.embeddings import EmbeddingBackend
from atlas_review.retrieval.keyword_index import (
    build_keyword_index,
    keyword_search,
    lexical_tokens,
    load_keyword_index,
    save_keyword_index,
)
from atlas_review.retrieval.vector_index import (
    build_dense_index,
    dense_search,
    restore_dense_index,
    save_dense_index,
)
from fakes import RecordingEmbedder

CONFIG = AppConfig()
TEXTS = (
    (
        "Technique AML.T0051.001 | LLM Prompt Injection / Indirect\n"
        "An adversary plants instructions in retrieved data."
    ),
    (
        "Case AML.CS0035 | Slack AI | Type: Exercise\n"
        "Indirect prompt injection exfiltrated workspace messages."
    ),
    "Mitigation AML.M0033 | Input Validation\nValidate untrusted text.",
)


def basis_vectors(count: int) -> np.ndarray:
    """Return unit basis vectors for ``count`` rows."""
    vectors = np.zeros((count, CONFIG.embedding.dimension), dtype=np.float32)
    for row in range(count):
        vectors[row, row] = 1.0
    return vectors


class WrongShapeEmbedder:
    """Return a vector of the wrong dimensionality."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``len(texts)`` three-dimensional rows."""
        return np.ones((len(list(texts)), 3), dtype=np.float32)


class NonFiniteEmbedder:
    """Return a vector containing NaN."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return NaN rows."""
        rows = np.ones((len(list(texts)), CONFIG.embedding.dimension), dtype=np.float32)
        rows[0, 0] = np.nan
        return rows


class SingleRowEmbedder:
    """Return one row regardless of the input count."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a single vector."""
        assert list(texts)
        return np.ones((1, CONFIG.embedding.dimension), dtype=np.float32)


class TestEmbeddings:
    """Queries get the BGE instruction once; passages never do."""

    def test_passage_embedding_never_applies_the_query_prefix(self) -> None:
        backend = RecordingEmbedder()
        vectors = embeddings.passage_vectors(backend, CONFIG.embedding, ["first"])
        assert backend.inputs == [("first",)]
        assert vectors.shape == (1, 768)

    def test_query_embedding_applies_the_instruction_exactly_once(self) -> None:
        backend = RecordingEmbedder()
        embeddings.query_vector(backend, CONFIG.embedding, "prompt injection")
        prefix = CONFIG.embedding.query_prefix
        assert backend.query_inputs == [f"{prefix}prompt injection"]

    def test_vectors_are_l2_normalized(self) -> None:
        backend = RecordingEmbedder()
        vectors = embeddings.passage_vectors(backend, CONFIG.embedding, list(TEXTS))
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        assert vectors.dtype == np.float32
        assert vectors.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(WrongShapeEmbedder(), id="shape"),
            pytest.param(NonFiniteEmbedder(), id="non-finite"),
        ],
    )
    def test_invalid_vectors_are_rejected(self, backend: EmbeddingBackend) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            embeddings.passage_vectors(backend, CONFIG.embedding, ["one"])
        assert caught.value.code == "invalid_embedding"

    def test_vector_count_must_match_the_input_count(self) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            embeddings.passage_vectors(
                SingleRowEmbedder(), CONFIG.embedding, ["a", "b"]
            )
        assert caught.value.code == "invalid_embedding"


class TestDenseIndex:
    """USearch keys are passage IDs, and save/restore keeps them."""

    def test_saved_index_restores_with_the_same_keys(self, tmp_path: Path) -> None:
        passage_ids = (7, 42, 1000)
        index = build_dense_index(basis_vectors(3), passage_ids, CONFIG.vector_index)
        path = tmp_path / "dense.usearch"
        save_dense_index(index, path)
        restored = restore_dense_index(path)
        assert restored.ndim == 768
        assert len(restored) == 3
        assert sorted(int(key) for key in restored.keys) == list(passage_ids)

    def test_search_returns_passage_ids_with_distances(self) -> None:
        passage_ids = (7, 42, 1000)
        index = build_dense_index(basis_vectors(3), passage_ids, CONFIG.vector_index)
        hits = dense_search(index, basis_vectors(1)[0], count=2, threads=1)
        assert len(hits) == 2
        assert hits[0][0] == 7
        assert {hit[0] for hit in hits} <= set(passage_ids)
        assert [distance for _, distance in hits] == sorted(
            distance for _, distance in hits
        )

    def test_empty_index_returns_no_hits(self) -> None:
        index = build_dense_index(
            np.zeros((0, 768), dtype=np.float32), (), CONFIG.vector_index
        )
        assert dense_search(index, basis_vectors(1)[0], count=5, threads=1) == ()

    def test_saved_index_is_a_file(self, tmp_path: Path) -> None:
        index = build_dense_index(basis_vectors(1), (5,), CONFIG.vector_index)
        path = tmp_path / "dense.usearch"
        save_dense_index(index, path)
        assert path.is_file()


class TestKeywordIndex:
    """The analyzer preserves ATLAS identifiers and ordinals resolve explicitly."""

    def test_complete_identifiers_survive_tokenization(self) -> None:
        tokens = lexical_tokens(
            ["Technique AML.T0051.001 and AML.TA0005", "AML.CS0035"],
            CONFIG.keyword_index,
        )
        assert "aml.t0051.001" in tokens[0]
        assert "aml.ta0005" in tokens[0]
        assert "aml.cs0035" in tokens[1]
        assert "t0051" not in tokens[0]

    def test_index_round_trip_preserves_ranking_and_ordinals(
        self, tmp_path: Path
    ) -> None:
        passage_ids = (7, 42, 1000)
        index = build_keyword_index(TEXTS, CONFIG.keyword_index)
        save_keyword_index(index, tmp_path / "keyword", CONFIG.keyword_index)
        loaded = load_keyword_index(tmp_path / "keyword")
        hits = keyword_search(
            loaded,
            "indirect prompt injection aml.cs0035",
            limit=3,
            passage_ids_by_ordinal=passage_ids,
            config=CONFIG.keyword_index,
        )
        assert hits[0][0] == 42
        assert hits[0][1] > 0.0
        assert all(score > 0.0 for _, score in hits)

    def test_out_of_vocabulary_query_returns_no_candidates(self) -> None:
        index = build_keyword_index(TEXTS, CONFIG.keyword_index)
        hits = keyword_search(
            index,
            "zzzz",
            limit=3,
            passage_ids_by_ordinal=(1, 2, 3),
            config=CONFIG.keyword_index,
        )
        assert hits == ()

    def test_empty_query_tokens_return_no_candidates(self) -> None:
        index = build_keyword_index(TEXTS, CONFIG.keyword_index)
        hits = keyword_search(
            index,
            "",
            limit=3,
            passage_ids_by_ordinal=(1, 2, 3),
            config=CONFIG.keyword_index,
        )
        assert hits == ()

    def test_ordinal_mapping_must_cover_the_index(self) -> None:
        index = build_keyword_index(TEXTS, CONFIG.keyword_index)
        with pytest.raises(AtlasReviewError) as caught:
            keyword_search(
                index,
                "prompt",
                limit=3,
                passage_ids_by_ordinal=(1, 2),
                config=CONFIG.keyword_index,
            )
        assert caught.value.code == "invalid_keyword_mapping"


class TestArtifactBundle:
    """Both indexes publish as one hashed, immutable bundle."""

    def test_bundle_contains_dense_keyword_mapping_and_manifest(
        self, tmp_path: Path
    ) -> None:
        bundle = _bundle(tmp_path)
        names = {name for name, _, _ in bundle.files}
        assert "dense.usearch" in names
        assert "passage_ids.msgpack" in names
        assert any(name.startswith("keyword/") for name in names)
        assert (bundle.directory / "artifacts.json").is_file()
        assert not list((tmp_path / "indexes").glob("*.tmp"))

    def test_manifest_hashes_every_generated_file(self, tmp_path: Path) -> None:
        bundle = _bundle(tmp_path)
        manifest = load_artifact_manifest(bundle.directory)
        listed = {entry["path"] for entry in manifest["files"]}
        for name, size, digest in bundle.files:
            assert name in listed
            path = bundle.directory / name
            assert path.stat().st_size == size
            assert digest == _digest(path)
        assert manifest["generation_id"] == "generation-1"
        assert manifest["passage_ids"] == [7, 42, 1000]

    def test_rebuilt_bundle_uses_a_new_directory(self, tmp_path: Path) -> None:
        first = _bundle(tmp_path, artifact_id="a")
        second = _bundle(tmp_path, artifact_id="b")
        assert first.directory != second.directory
        assert first.directory.is_dir()
        assert second.directory.is_dir()

    def test_repeated_publication_replaces_an_orphan_bundle(
        self, tmp_path: Path
    ) -> None:
        first = _bundle(tmp_path, artifact_id="a")
        second = _bundle(tmp_path, artifact_id="a")
        assert second.directory == first.directory
        assert (second.directory / "artifacts.json").is_file()
        assert not list((tmp_path / "indexes").glob("*.tmp"))

    def test_mismatched_populations_are_rejected(self, tmp_path: Path) -> None:
        indexes = tmp_path / "indexes"
        indexes.mkdir()
        with pytest.raises(AtlasReviewError) as caught:
            build_bundle(
                publication=Publication(
                    indexes_dir=indexes,
                    generation_id="generation-1",
                    artifact_id="a",
                ),
                vectors=basis_vectors(3),
                passage_ids=(7, 42),
                retrieval_texts=TEXTS,
                config=CONFIG,
            )
        assert caught.value.code == "invalid_index_inputs"
        assert not any(indexes.iterdir())


def _bundle(tmp_path: Path, artifact_id: str = "a") -> ArtifactBundle:
    """Build the small fixture bundle used by these tests."""
    indexes = tmp_path / "indexes"
    indexes.mkdir(exist_ok=True)
    return build_bundle(
        publication=Publication(
            indexes_dir=indexes,
            generation_id="generation-1",
            artifact_id=artifact_id,
        ),
        vectors=basis_vectors(3),
        passage_ids=(7, 42, 1000),
        retrieval_texts=TEXTS,
        config=CONFIG,
    )


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
