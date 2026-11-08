"""Stored application configuration (schema version 2).

The whole configuration is one validated Pydantic model. Fixed requirements
from the specification are enforced by validators rather than by UI choices.
"""

from __future__ import annotations

import ipaddress
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from atlas_review.contracts import AppModel, RequiredText, VersionTwo

PINNED_SOURCE_URL = (
    "https://raw.githubusercontent.com/mitre-atlas/atlas-data/"
    "3259f388d19cbcca11bacf12a0ef97f4198f711b/dist/v6/ATLAS-2026.09.yaml"
)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"
EMBEDDING_DIMENSION = 768
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ATLAS_TOKEN_PATTERN = r"(?u)\baml\.(?:ta|cs|t|m)\d{4}(?:\.\d{3})?\b|\b\w+\b"
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_REQUEST_CHARACTERS = 40000
SUPPORTED_FORMAT_MAJOR = 6
PLACEHOLDER_BASE_URL = "https://your-provider.example/v1"
PLACEHOLDER_MODEL = "your-model-id"


class CorpusConfig(AppModel):
    """ATLAS corpus source location and download limits."""

    default_url: RequiredText = PINNED_SOURCE_URL
    supported_format_major: int = SUPPORTED_FORMAT_MAJOR
    download_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_source_bytes: int = Field(default=MAX_SOURCE_BYTES, gt=0, le=MAX_SOURCE_BYTES)

    @field_validator("default_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"https", "http"} or not parts.hostname:
            raise ValueError("default_url must be an absolute HTTP(S) URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("default_url must not embed credentials")
        return value

    @field_validator("supported_format_major")
    @classmethod
    def require_supported_major(cls, value: int) -> int:
        if value != SUPPORTED_FORMAT_MAJOR:
            raise ValueError("only ATLAS format major 6 is supported")
        return value


class EmbeddingConfig(AppModel):
    """FastEmbed embedding model settings fixed by the specification."""

    model_name: RequiredText = EMBEDDING_MODEL
    dimension: int = EMBEDDING_DIMENSION
    query_prefix: RequiredText = BGE_QUERY_PREFIX
    batch_size: int = Field(default=32, gt=0)
    threads: int = Field(default=4, ge=1)
    provider: RequiredText = "CPUExecutionProvider"

    @field_validator("model_name")
    @classmethod
    def require_fixed_model(cls, value: str) -> str:
        if value != EMBEDDING_MODEL:
            raise ValueError(f"embedding model is fixed at {EMBEDDING_MODEL}")
        return value

    @field_validator("dimension")
    @classmethod
    def require_fixed_dimension(cls, value: int) -> int:
        if value != EMBEDDING_DIMENSION:
            raise ValueError(f"embedding dimension is fixed at {EMBEDDING_DIMENSION}")
        return value

    @field_validator("provider")
    @classmethod
    def require_cpu_provider(cls, value: str) -> str:
        if value != "CPUExecutionProvider":
            raise ValueError("only CPUExecutionProvider is supported in this release")
        return value


class ChunkingConfig(AppModel):
    """Passage window budgets applied by the chunker."""

    body_target_tokens: int = Field(default=192, gt=0)
    passage_max_tokens: int = Field(default=352, gt=0)
    overlap_tokens: int = Field(default=32, ge=0)
    header_max_tokens: int = Field(default=40, gt=0)
    body_max_tokens: int = Field(default=192, gt=0)

    @model_validator(mode="after")
    def check_body_budget(self) -> Self:
        if self.body_target_tokens > self.body_max_tokens:
            raise ValueError("body_target_tokens must not exceed body_max_tokens")
        if self.overlap_tokens >= self.body_target_tokens:
            raise ValueError("overlap_tokens must be smaller than body_target_tokens")
        if self.passage_max_tokens <= self.header_max_tokens + self.body_max_tokens:
            raise ValueError(
                "passage_max_tokens must leave room for header and body budgets"
            )
        return self


class ContextualizationConfig(AppModel):
    """Generated chunk-context request settings."""

    enabled: bool = True
    document_builder_version: RequiredText = "atlas-owner-v1"
    prompt_version: RequiredText = "chunk-context-v1"
    context_max_tokens: int = Field(default=96, gt=0)
    output_tokens: int = Field(default=256, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_input_characters: int = Field(
        default=MAX_REQUEST_CHARACTERS, gt=0, le=MAX_REQUEST_CHARACTERS
    )
    concurrency: int = 1
    automatic_retries: int = 0
    cache_policy: Literal["reuse_valid"] = "reuse_valid"
    regeneration_revision: int = Field(default=0, ge=0)

    @field_validator("enabled")
    @classmethod
    def require_contextualization(cls, value: bool) -> bool:
        if not value:
            raise ValueError("contextualization.enabled must be true in this release")
        return value

    @field_validator("concurrency")
    @classmethod
    def require_sequential_requests(cls, value: int) -> int:
        if value != 1:
            raise ValueError("contextualization is sequential; concurrency must be 1")
        return value

    @field_validator("automatic_retries")
    @classmethod
    def require_no_retries(cls, value: int) -> int:
        if value != 0:
            raise ValueError("automatic retries must be 0; retries are explicit")
        return value


class VectorIndexConfig(AppModel):
    """USearch dense index parameters."""

    metric: Literal["cos"] = "cos"
    dtype: Literal["f32"] = "f32"
    connectivity: int = Field(default=16, gt=0)
    expansion_add: int = Field(default=128, gt=0)
    expansion_search: int = Field(default=64, gt=0)
    threads: int = Field(default=4, ge=1)


class KeywordIndexConfig(AppModel):
    """bm25s keyword index parameters."""

    implementation: Literal["bm25s"] = "bm25s"
    method: Literal["lucene"] = "lucene"
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    backend: Literal["numpy"] = "numpy"
    dtype: Literal["float32"] = "float32"
    lowercase: bool = True
    stopwords: list[str] = Field(default_factory=list)
    stemming: bool = False
    analyzer_version: RequiredText = "atlas-lexical-v1"
    token_pattern: RequiredText = ATLAS_TOKEN_PATTERN

    @field_validator("stopwords")
    @classmethod
    def require_no_stopwords(cls, value: list[str]) -> list[str]:
        if value:
            raise ValueError("stopwords must be empty; the analyzer removes no words")
        return value

    @field_validator("stemming")
    @classmethod
    def require_no_stemming(cls, value: bool) -> bool:
        if value:
            raise ValueError("stemming must be disabled in this release")
        return value

    @field_validator("lowercase")
    @classmethod
    def require_lowercase(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "lowercase must stay enabled so indexing and queries match"
            )
        return value


MAX_RERANK_CANDIDATES = 128


class RetrievalConfig(AppModel):
    """Retrieval candidate budgets for fusion and reranking."""

    query_max_tokens: int = Field(default=128, gt=0)
    dense_k: int = Field(default=32, gt=0)
    exact_id_limit: int = Field(default=4, gt=0)
    seed_entity_limit: int = Field(default=8, gt=0)
    graph_max_hops: int = Field(default=2, ge=0, le=2)
    graph_max_nodes: int = Field(default=96, gt=0)
    graph_max_edges: int = Field(default=256, gt=0)
    graph_max_edges_per_node: int = Field(default=24, gt=0)
    graph_passage_limit: int = Field(default=64, gt=0)
    rerank_candidate_limit: int = Field(default=96, gt=0)
    selected_passage_limit: int = Field(default=12, gt=0)
    per_owner_passage_limit: int = Field(default=2, gt=0)
    evidence_item_limit: int = Field(default=24, gt=0)
    evidence_character_limit: int = Field(default=20000, gt=0)
    bm25_k: int = Field(default=32, gt=0)
    fusion_rrf_k: int = Field(default=60, gt=0)
    direct_candidate_limit: int = Field(default=48, gt=0)

    @model_validator(mode="after")
    def check_candidate_budgets(self) -> Self:
        if self.rerank_candidate_limit > MAX_RERANK_CANDIDATES:
            raise ValueError("rerank_candidate_limit must not exceed 128")
        if self.direct_candidate_limit < self.exact_id_limit:
            raise ValueError("direct_candidate_limit must be at least exact_id_limit")
        if self.direct_candidate_limit >= self.rerank_candidate_limit:
            raise ValueError(
                "direct_candidate_limit must be strictly less than rerank_candidate_limit"
            )
        if self.evidence_item_limit < self.selected_passage_limit:
            raise ValueError(
                "evidence_item_limit must be at least selected_passage_limit"
            )
        if self.exact_id_limit > self.seed_entity_limit:
            raise ValueError("exact_id_limit must not exceed seed_entity_limit")
        return self


class RerankerConfig(AppModel):
    """FastEmbed cross-encoder reranker settings."""

    model_name: RequiredText = RERANKER_MODEL
    batch_size: int = Field(default=16, gt=0)
    threads: int = Field(default=4, ge=1)
    provider: RequiredText = "CPUExecutionProvider"

    @field_validator("model_name")
    @classmethod
    def require_fixed_model(cls, value: str) -> str:
        if value != RERANKER_MODEL:
            raise ValueError(f"reranker model is fixed at {RERANKER_MODEL}")
        return value


class LLMConfig(AppModel):
    """OpenAI-compatible provider settings; the API key is never stored here."""

    provider_name: RequiredText = "custom-compatible"
    base_url: RequiredText
    model: RequiredText
    api_key_mode: Literal["environment", "none"] = "environment"
    api_key_env: str | None = "ATLAS_REVIEW_API_KEY"
    timeout_seconds: float = Field(default=90.0, gt=0)
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = (
        "max_tokens"
    )
    output_tokens: int = Field(default=2500, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    response_format_mode: Literal["text", "json_object"] = "text"
    max_input_characters: int = Field(
        default=MAX_REQUEST_CHARACTERS, gt=0, le=MAX_REQUEST_CHARACTERS
    )
    context_window_tokens: int | None = Field(default=None, gt=0)
    max_findings: int = Field(default=5, gt=0)

    @field_validator("provider_name", "base_url", "model")
    @classmethod
    def nonblank_setting(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider settings must not be blank")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_api_root(cls, value: str) -> str:
        parts = urlsplit(value)
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) API root without credentials, query, or fragment"
            )
        if parts.scheme == "http":
            host = parts.hostname
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host.lower() == "localhost"
            if not loopback:
                raise ValueError("non-loopback API endpoints require HTTPS")
        return value

    @model_validator(mode="after")
    def valid_credential_reference(self) -> Self:
        if self.api_key_mode == "environment":
            if self.api_key_env is None or not _is_environment_name(self.api_key_env):
                raise ValueError(
                    "environment authentication requires a valid environment-variable name"
                )
        elif self.api_key_env is not None:
            raise ValueError("unauthenticated mode requires api_key_env to be null")
        return self


def _is_environment_name(value: str) -> bool:
    if not value or not (value[0].isalpha() or value[0] == "_"):
        return False
    return all(character.isalnum() or character == "_" for character in value)


class UiConfig(AppModel):
    """Terminal display preferences."""

    page_size: int = Field(default=10, gt=0)
    color: Literal["auto", "always", "never"] = "auto"


class AppConfig(AppModel):
    """Complete stored workspace configuration."""

    schema_version: VersionTwo = 2
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    contextualization: ContextualizationConfig = Field(
        default_factory=ContextualizationConfig
    )
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    keyword_index: KeywordIndexConfig = Field(default_factory=KeywordIndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(
        default_factory=lambda: LLMConfig(
            base_url=PLACEHOLDER_BASE_URL, model=PLACEHOLDER_MODEL
        )
    )
    ui: UiConfig = Field(default_factory=UiConfig)

    @model_validator(mode="after")
    def check_passage_budget(self) -> Self:
        reserved = (
            self.chunking.header_max_tokens
            + self.chunking.body_max_tokens
            + self.contextualization.context_max_tokens
        )
        if self.chunking.passage_max_tokens <= reserved:
            raise ValueError(
                "passage_max_tokens must exceed the header, body, and generated-context budgets"
            )
        return self


def config_from_json(raw: str) -> AppConfig:
    """Validate a full application configuration file."""
    return AppConfig.model_validate_json(raw, strict=True)


def config_to_json(config: AppConfig) -> str:
    """Serialize a validated configuration for import/export."""
    return config.model_dump_json(indent=2)


def apply_provider_settings(config: AppConfig, raw: str) -> AppConfig:
    """Replace only ``llm`` after validating the provider file and whole config."""
    provider = LLMConfig.model_validate_json(raw, strict=True)
    return AppConfig.model_validate(
        {**config.model_dump(mode="json"), "llm": provider.model_dump(mode="json")},
        strict=True,
    )
