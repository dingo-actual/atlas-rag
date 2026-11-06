"""Configuration tests: defaults, fixed requirements, cross-field rules, JSON round trips."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from atlas_review.config import (
    AppConfig,
    apply_provider_settings,
    config_from_json,
    config_to_json,
)

PROVIDER_JSON = json.dumps(
    {
        "provider_name": "my-compatible-provider",
        "base_url": "https://provider.example/v1",
        "model": "model-id",
        "api_key_mode": "environment",
        "api_key_env": "MY_PROVIDER_API_KEY",
    }
)


def config_with(section: str | None, changes: dict[str, Any]) -> dict[str, Any]:
    data = AppConfig().model_dump(mode="json")
    if section is None:
        data.update(changes)
    else:
        data[section].update(changes)
    return data


def test_default_config_satisfies_fixed_requirements() -> None:
    config = AppConfig()
    assert config.schema_version == 2
    assert config.corpus.supported_format_major == 6
    assert "3259f388d19cbcca11bacf12a0ef97f4198f711b" in config.corpus.default_url
    assert config.embedding.model_name == "BAAI/bge-base-en-v1.5"
    assert config.embedding.dimension == 768
    assert config.vector_index.metric == "cos"
    assert config.vector_index.dtype == "f32"
    assert config.keyword_index.implementation == "bm25s"
    assert config.keyword_index.method == "lucene"
    assert config.reranker.model_name == "Xenova/ms-marco-MiniLM-L-12-v2"
    assert config.contextualization.enabled is True
    assert config.retrieval.fusion_rrf_k == 60
    assert config.llm.api_key_env == "ATLAS_REVIEW_API_KEY"


def test_config_json_round_trip() -> None:
    config = AppConfig()
    restored = config_from_json(config_to_json(config))
    assert restored == config
    assert (
        json.loads(config_to_json(config))["llm"]["api_key_env"]
        == "ATLAS_REVIEW_API_KEY"
    )


@pytest.mark.parametrize(
    ("section", "changes", "match"),
    [
        pytest.param(None, {"schema_version": 3}, "2", id="config-version"),
        pytest.param(None, {"schema_version": True}, "integer", id="boolean-version"),
        pytest.param(
            "corpus", {"supported_format_major": 5}, "format major", id="format-major"
        ),
        pytest.param(
            "corpus", {"max_source_bytes": 21 * 1024 * 1024}, "20971520", id="size-cap"
        ),
        pytest.param(
            "embedding", {"model_name": "other/model"}, "bge-base", id="embedding-model"
        ),
        pytest.param("embedding", {"dimension": 384}, "768", id="dimension"),
        pytest.param("embedding", {"batch_size": 0}, "batch_size", id="batch-size"),
        pytest.param("vector_index", {"metric": "l2"}, "cos", id="metric"),
        pytest.param("vector_index", {"dtype": "f16"}, "f32", id="vector-dtype"),
        pytest.param("chunking", {"overlap_tokens": 192}, "overlap", id="overlap"),
        pytest.param(
            "chunking",
            {"body_target_tokens": 300},
            "body_target_tokens",
            id="target-above-max",
        ),
        pytest.param(
            "contextualization", {"enabled": False}, "enabled", id="context-required"
        ),
        pytest.param(
            "contextualization", {"concurrency": 4}, "sequential", id="concurrency"
        ),
        pytest.param(
            "contextualization", {"automatic_retries": 2}, "retries", id="retries"
        ),
        pytest.param(
            "keyword_index", {"method": "robertson"}, "lucene", id="bm25-method"
        ),
        pytest.param("keyword_index", {"k1": 0.0}, "k1", id="k1"),
        pytest.param("keyword_index", {"b": 1.5}, "less than or equal to 1", id="b"),
        pytest.param("keyword_index", {"stemming": True}, "stemming", id="stemming"),
        pytest.param(
            "keyword_index", {"stopwords": ["the"]}, "stopwords", id="stopwords"
        ),
        pytest.param("retrieval", {"graph_max_hops": 3}, "graph_max_hops", id="hops"),
        pytest.param(
            "retrieval",
            {"rerank_candidate_limit": 256},
            "rerank_candidate_limit",
            id="rerank-cap",
        ),
        pytest.param(
            "retrieval",
            {"direct_candidate_limit": 96},
            "direct_candidate_limit",
            id="direct-not-below-rerank",
        ),
        pytest.param(
            "retrieval",
            {"exact_id_limit": 64},
            "exact_id_limit",
            id="exact-above-direct",
        ),
        pytest.param(
            "retrieval",
            {"evidence_item_limit": 4},
            "evidence_item_limit",
            id="evidence-below-passages",
        ),
        pytest.param("retrieval", {"fusion_rrf_k": 0}, "fusion_rrf_k", id="rrf"),
        pytest.param("retrieval", {"dense_k": 0}, "dense_k", id="dense-k"),
        pytest.param("retrieval", {"bm25_k": 0}, "bm25_k", id="bm25-k"),
        pytest.param(
            "llm", {"base_url": "http://example.com/v1"}, "HTTPS", id="plain-http"
        ),
        pytest.param(
            "llm",
            {"base_url": "https://user:secret@example.com/v1"},
            "credentials",
            id="userinfo",
        ),
        pytest.param(
            "llm", {"base_url": "https://example.com/v1?key=1"}, "query", id="query"
        ),
        pytest.param(
            "llm",
            {"api_key_mode": "none"},
            "api_key_env",
            id="unauthenticated-with-env-name",
        ),
        pytest.param(
            "llm",
            {"api_key_env": "1BAD"},
            "environment-variable",
            id="bad-env-name",
        ),
        pytest.param("llm", {"output_tokens": 0}, "output_tokens", id="output-tokens"),
        pytest.param("llm", {"timeout_seconds": 0.0}, "timeout_seconds", id="timeout"),
        pytest.param("ui", {"page_size": 0}, "page_size", id="page-size"),
    ],
)
def test_config_rejects_invalid_settings(
    section: str | None, changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        AppConfig.model_validate(config_with(section, changes), strict=True)


def test_config_accepts_loopback_http_and_unauthenticated_mode() -> None:
    config = AppConfig.model_validate(
        config_with(
            "llm",
            {
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key_mode": "none",
                "api_key_env": None,
            },
        ),
        strict=True,
    )
    assert config.llm.api_key_mode == "none"
    assert config.llm.api_key_env is None


def test_config_rejects_unknown_section_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AppConfig.model_validate(
            config_with("retrieval", {"additional": 1}), strict=True
        )


def test_provider_settings_replace_only_the_llm_section() -> None:
    current = AppConfig()
    replaced = apply_provider_settings(current, PROVIDER_JSON)
    assert replaced.llm.base_url == "https://provider.example/v1"
    assert replaced.llm.api_key_env == "MY_PROVIDER_API_KEY"
    assert replaced.retrieval == current.retrieval
    assert replaced.embedding == current.embedding


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("{not json", id="malformed-json"),
        pytest.param(
            json.dumps({"base_url": "https://provider.example/v1"}), id="missing-fields"
        ),
        pytest.param(
            json.dumps(
                {"base_url": "https://provider.example/v1", "model": "m", "extra": 1}
            ),
            id="unknown-field",
        ),
        pytest.param(
            json.dumps({"base_url": "http://provider.example/v1", "model": "m"}),
            id="plain-http",
        ),
    ],
)
def test_provider_settings_import_rejects_invalid_files(raw: str) -> None:
    with pytest.raises(ValidationError):
        apply_provider_settings(AppConfig(), raw)
