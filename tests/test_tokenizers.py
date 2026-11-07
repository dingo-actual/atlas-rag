"""Tokenizer adapter: FastEmbed description conversion and artifact loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastembed import TextEmbedding
from fastembed.common.model_description import BaseModelDescription
from fastembed.rerank.cross_encoder import TextCrossEncoder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from atlas_review.config import EMBEDDING_MODEL, RERANKER_MODEL
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.tokenizers import (
    TOKENIZER_FILE,
    ModelTokenCounter,
    load_model_token_counter,
)

WORD_LEVEL_VOCAB = {"hello": 0, "world": 1, "indirect": 2, "injection": 3, "[UNK]": 4}
DIGEST_SUFFIX = re.compile(r"@[0-9a-f]{16}$")


def write_word_level_tokenizer(directory: Path) -> None:
    """Write a real, tiny tokenizer artifact into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(WORD_LEVEL_VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / TOKENIZER_FILE))


@pytest.fixture
def artifact_directory(tmp_path: Path) -> Path:
    """Return a model directory holding a real tokenizer artifact."""
    directory = tmp_path / "artifact"
    write_word_level_tokenizer(directory)
    return directory


class TestDescriptionConversion:
    """The adapter converts FastEmbed's public model listing for its downloader."""

    @pytest.mark.parametrize(
        ("model_class", "model_name", "expected_source"),
        [
            pytest.param(
                TextEmbedding,
                EMBEDDING_MODEL,
                "Qdrant/bge-base-en-v1.5-onnx-Q",
                id="embedding",
            ),
            pytest.param(
                TextCrossEncoder, RERANKER_MODEL, RERANKER_MODEL, id="reranker"
            ),
        ],
    )
    def test_description_keeps_the_resolved_artifact_source(
        self,
        model_class: type[TextEmbedding] | type[TextCrossEncoder],
        model_name: str,
        expected_source: str,
        artifact_directory: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[BaseModelDescription] = []

        def fake_download(
            description: BaseModelDescription, _cache_dir: str, **_: Any
        ) -> Path:
            seen.append(description)
            return artifact_directory

        monkeypatch.setattr(model_class, "download_model", fake_download)
        counter = load_model_token_counter(model_class, model_name, artifact_directory)
        assert counter.name.startswith(f"{model_name}@")
        assert DIGEST_SUFFIX.search(counter.name)
        assert seen[0].model == model_name
        assert seen[0].sources.hf == expected_source
        assert seen[0].model_file


class TestCounterMeasurement:
    """Counts come from the artifact tokenizer with truncation disabled."""

    def test_tokens_are_counted_and_pairs_are_supported(
        self, artifact_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            TextEmbedding,
            "download_model",
            lambda _description, _cache_dir, **_: artifact_directory,
        )
        counter = load_model_token_counter(
            TextEmbedding, EMBEDDING_MODEL, artifact_directory
        )
        assert isinstance(counter, ModelTokenCounter)
        assert counter.count("hello world") == 2
        assert counter.count_pair("indirect", "injection") == 2

    def test_offsets_map_to_characters(
        self, artifact_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            TextEmbedding,
            "download_model",
            lambda _description, _cache_dir, **_: artifact_directory,
        )
        counter = load_model_token_counter(
            TextEmbedding, EMBEDDING_MODEL, artifact_directory
        )
        text = "hello world indirect"
        assert counter.prefix_offset(text, 2) == len("hello world")
        assert counter.tail_offset(text, 1) == text.index("indirect")


class TestAcquisitionFailures:
    """Unsupported models and incomplete artifacts fail with a clear code."""

    def test_unsupported_model_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            load_model_token_counter(TextEmbedding, "not/a-model", tmp_path)
        assert caught.value.code == "unsupported_model"
        assert caught.value.stage == "model_acquisition"

    def test_missing_tokenizer_file_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            TextEmbedding,
            "download_model",
            lambda _description, _cache_dir, **_: tmp_path,
        )
        with pytest.raises(AtlasReviewError) as caught:
            load_model_token_counter(TextEmbedding, EMBEDDING_MODEL, tmp_path)
        assert caught.value.code == "missing_tokenizer"
