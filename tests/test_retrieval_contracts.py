"""Contracts for contextualized passages, cache records, and search candidates."""

from typing import Any

import pytest
from pydantic import ValidationError

from atlas_review.contracts import (
    Candidate,
    ChunkContext,
    ContextDocument,
    ContextualizationResult,
    GenerationManifest,
    Passage,
)

SHA = "a" * 64

VALID_DOCUMENT: dict[str, Any] = {
    "schema_version": 1,
    "owner_id": "AML.CS0035",
    "owner_kind": "case_study",
    "document": '{"owner": "AML.CS0035"}',
    "support_locators": ["/case-studies/AML.CS0035/description"],
    "document_sha256": SHA,
}

VALID_CONTEXT: dict[str, Any] = {
    "schema_version": 1,
    "cache_key": SHA,
    "prompt_version": "chunk-context-v1",
    "model": "fake-model-1",
    "response_model": "fake-model-1",
    "response_id": "resp-1",
    "context_text": "The excerpt describes the planted instructions.",
    "support_locators": ["/case-studies/AML.CS0035/description"],
    "embedding_tokens": 20,
    "reranker_tokens": 22,
    "input_tokens": 900,
    "output_tokens": 25,
    "duration_seconds": 0.5,
    "request_fingerprint": SHA,
    "created_at": "2026-11-01T00:00:00+00:00",
}

VALID_PASSAGE: dict[str, Any] = {
    "schema_version": 1,
    "passage_id": 7,
    "generation_id": "generation-1",
    "entity_id": "AML.T0051.001",
    "owner_kind": "technique",
    "section": "description",
    "ordinal": 0,
    "bm25_ordinal": 0,
    "header": "Technique AML.T0051.001 | LLM Prompt Injection / Indirect",
    "source_text": "An adversary may place instructions in retrieved data.",
    "context_text": "The excerpt explains indirect prompt injection.",
    "retrieval_text": "Technique AML.T0051.001 | title\nContext (generated):\nnote",
    "context_cache_key": SHA,
    "source_sha256": SHA,
    "content_sha256": SHA,
    "embedding_tokens": 40,
    "reranker_tokens": 42,
    "source_locator": "/techniques/AML.T0051.001/description",
    "source_start": 0,
    "source_end": 48,
}

VALID_CANDIDATE: dict[str, Any] = {
    "passage_id": 7,
    "channels": ["dense", "bm25"],
    "dense_rank": 1,
    "dense_distance": 0.25,
    "bm25_rank": 3,
    "bm25_score": 1.5,
    "fusion_score": 0.032,
}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"support_locators": []}, "support_locators", id="no-locators"),
        pytest.param(
            {"support_locators": ["/a"] * 5}, "support_locators", id="too-many-locators"
        ),
        pytest.param({"context": "   "}, "whitespace", id="blank-context"),
        pytest.param({"extra": "x"}, "extra_forbidden", id="extra-field"),
    ],
)
def test_chunk_context_rejects_unusable_output(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        ChunkContext.model_validate(
            {"context": "note", "support_locators": ["/a"], **overrides}
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"owner_kind": "tactic"}, "owner_kind", id="unsupported-owner"),
        pytest.param({"document_sha256": "short"}, "document_sha256", id="bad-hash"),
        pytest.param({"support_locators": []}, "support_locators", id="no-locators"),
        pytest.param(
            {"support_locators": ["relative/path"]}, "locator", id="relative-locator"
        ),
    ],
)
def test_context_document_rejects_invalid_records(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        ContextDocument.model_validate({**VALID_DOCUMENT, **overrides}, strict=True)


def test_contextualization_result_keeps_missing_usage_as_null() -> None:
    record = ContextualizationResult.model_validate(
        {**VALID_CONTEXT, "input_tokens": None, "output_tokens": None}, strict=True
    )
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.support_locators == ["/case-studies/AML.CS0035/description"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"embedding_tokens": 0}, "embedding_tokens", id="zero-tokens"),
        pytest.param({"duration_seconds": -1.0}, "duration_seconds", id="negative"),
        pytest.param({"response_id": ""}, "response_id", id="blank-response-id"),
    ],
)
def test_contextualization_result_rejects_invalid_usage(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        ContextualizationResult.model_validate(
            {**VALID_CONTEXT, **overrides}, strict=True
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param(
            {"entity_id": "AML.T0051.001", "edge_id": "b" * 64},
            "exactly one owner",
            id="two-owners",
        ),
        pytest.param({"entity_id": None}, "exactly one owner", id="no-owner"),
        pytest.param({"source_end": 0}, "source_end", id="empty-source-range"),
        pytest.param({"passage_id": 0}, "passage_id", id="zero-passage-id"),
        pytest.param({"embedding_tokens": 0}, "embedding_tokens", id="zero-tokens"),
        pytest.param({"owner_kind": "tactic"}, "owner_kind", id="unsupported-owner"),
    ],
)
def test_passage_rejects_invalid_records(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Passage.model_validate({**VALID_PASSAGE, **overrides}, strict=True)


def test_edge_owned_passage_accepts_an_edge_owner() -> None:
    passage = Passage.model_validate(
        {
            **VALID_PASSAGE,
            "entity_id": None,
            "edge_id": "b" * 64,
            "owner_kind": "case_study",
            "section": "employs",
        },
        strict=True,
    )
    assert passage.entity_id is None
    assert passage.edge_id == "b" * 64


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"channels": []}, "channels", id="no-channel"),
        pytest.param({"channels": ["dense", "dense"]}, "channels", id="duplicates"),
        pytest.param({"channels": ["ranked"]}, "channels", id="unknown-channel"),
        pytest.param({"dense_rank": 0}, "dense_rank", id="zero-rank"),
        pytest.param({"fusion_score": -0.1}, "fusion_score", id="negative-fusion"),
    ],
)
def test_candidate_rejects_invalid_channel_records(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        Candidate.model_validate({**VALID_CANDIDATE, **overrides}, strict=True)


def test_candidate_allows_absent_channel_evidence() -> None:
    candidate = Candidate.model_validate(
        {
            "passage_id": 9,
            "channels": ["bm25"],
            "bm25_rank": 1,
            "bm25_score": 2.0,
            "fusion_score": 0.016,
        },
        strict=True,
    )
    assert candidate.dense_rank is None
    assert candidate.dense_distance is None


def test_generation_manifest_defaults_stay_loadable_for_increment_one_payloads() -> (
    None
):
    manifest = GenerationManifest.model_validate(
        {
            "schema_version": 1,
            "generation_id": "generation-1",
            "created_at": "2026-11-01T00:00:00+00:00",
            "pipeline_version": "atlas-ingest-v1",
            "build_fingerprint": SHA,
            "snapshot": {
                "origin": "fixture",
                "format_version": "6.0.0",
                "collection_version": "2026.09",
                "acquired_at": "2026-11-01T00:00:00+00:00",
                "source_sha256": SHA,
            },
            "entity_count": 0,
            "edge_count": 0,
        },
        strict=True,
    )
    assert manifest.passage_count == 0
    assert manifest.embedding is None
    assert manifest.contextualization is None
