"""Pydantic v2 contracts and the strict validation policy they share.

Pydantic is the single schema authority. Storage payloads are plain values
produced by ``model_dump(mode="json")`` and revalidated when decoded.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

type JsonValue = (
    str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
)

_BRIEF_CHARACTER_LIMIT = 12000
_JSON_INTEGER_MESSAGE = "schema_version must be a JSON integer"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHA256 = re.compile(_SHA256_PATTERN)


def require_json_integer(value: object) -> object:
    """Reject ``True`` and ``1.0`` where a JSON integer version is required."""
    if type(value) is not int:
        raise ValueError(_JSON_INTEGER_MESSAGE)
    return value


VersionOne = Annotated[Literal[1], BeforeValidator(require_json_integer)]
VersionTwo = Annotated[Literal[2], BeforeValidator(require_json_integer)]
RequiredText = Annotated[str, Field(min_length=1)]
ShortText = Annotated[str, Field(min_length=1, max_length=1000)]
Explanation = Annotated[str, Field(min_length=1, max_length=2000)]
BriefItems = Annotated[list[ShortText], Field(max_length=20)]
Sha256Hex = Annotated[str, Field(pattern=_SHA256_PATTERN)]
OwnerKind = Literal["technique", "mitigation", "case_study"]


class AppModel(BaseModel):
    """Strict, frozen base for every application-owned contract."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


def _require_absolute_locator(value: str) -> str:
    if not value.startswith("/"):
        raise ValueError("source_locator must be an absolute source path")
    return value


def _require_nonblank_optional(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("optional text must be null or non-blank")
    return value


class ReviewRequest(AppModel):
    """Application brief supplied by the reviewer (spec section 3.2)."""

    schema_version: VersionOne
    title: RequiredText
    review_question: RequiredText
    retrieval_focus: RequiredText
    system_description: RequiredText
    users_and_roles: BriefItems = Field(default_factory=list)
    data_sources: BriefItems = Field(default_factory=list)
    untrusted_inputs: BriefItems = Field(default_factory=list)
    tools_and_permissions: BriefItems = Field(default_factory=list)
    sensitive_assets: BriefItems = Field(default_factory=list)
    existing_controls: BriefItems = Field(default_factory=list)
    constraints: BriefItems = Field(default_factory=list)
    unknowns: BriefItems = Field(default_factory=list)

    @model_validator(mode="after")
    def check_text_budget(self) -> Self:
        total = sum(
            len(value)
            for value in (
                self.title,
                self.review_question,
                self.retrieval_focus,
                self.system_description,
            )
        )
        for values in (
            self.users_and_roles,
            self.data_sources,
            self.untrusted_inputs,
            self.tools_and_permissions,
            self.sensitive_assets,
            self.existing_controls,
            self.constraints,
            self.unknowns,
        ):
            total += sum(len(value) for value in values)
        if total > _BRIEF_CHARACTER_LIMIT:
            raise ValueError("brief text exceeds the 12000-character limit")
        return self


class SourceReference(AppModel):
    """Bibliography entry owned by one ATLAS record."""

    ref_id: RequiredText
    title: RequiredText
    url: str | None = None

    @field_validator("url")
    @classmethod
    def require_http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("https://", "http://")):
            raise ValueError("reference URL must be an ordinary HTTP(S) link")
        return value


class SourceSnapshot(AppModel):
    """Exact identity of the bytes a generation was built from."""

    origin: RequiredText
    format_version: RequiredText
    collection_version: RequiredText
    acquired_at: RequiredText
    source_sha256: str
    commit: str | None = None
    license: str | None = None

    @field_validator("source_sha256")
    @classmethod
    def require_lowercase_digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("origin", "format_version", "collection_version")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source identity text must not be blank")
        return value

    @field_validator("acquired_at")
    @classmethod
    def require_utc_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("acquired_at must be an ISO 8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError(
                "acquired_at must be an ISO 8601 timestamp with a UTC offset"
            )
        return value


class AtlasEntity(AppModel):
    """One normalized ATLAS record of any supported kind."""

    generation_id: RequiredText
    entity_id: RequiredText
    kind: Literal[
        "collection", "matrix", "tactic", "technique", "mitigation", "case_study"
    ]
    name: RequiredText
    display_name: RequiredText
    description: RequiredText
    source_locator: RequiredText
    references: list[SourceReference] = Field(default_factory=list)
    source_fields: dict[str, JsonValue] = Field(default_factory=dict)
    uuid: str | None = None
    created_date: str | None = None
    modified_date: str | None = None
    platforms: list[str] = Field(default_factory=list)
    maturity: str | None = None
    lifecycle_phases: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    case_type: str | None = None
    case_actor: str | None = None
    case_target: str | None = None
    case_date: str | None = None
    case_date_granularity: str | None = None
    attack_reference: str | None = None
    reporter: str | None = None

    @field_validator("source_locator")
    @classmethod
    def require_locator(cls, value: str) -> str:
        return _require_absolute_locator(value)

    @field_validator(
        "uuid",
        "created_date",
        "modified_date",
        "maturity",
        "case_type",
        "case_actor",
        "case_target",
        "case_date",
        "case_date_granularity",
        "attack_reference",
        "reporter",
    )
    @classmethod
    def require_nonblank_optional_text(cls, value: str | None) -> str | None:
        return _require_nonblank_optional(value)


class AtlasRelationship(AppModel):
    """One stored ATLAS relationship edge, with its source orientation."""

    generation_id: RequiredText
    edge_id: Sha256Hex
    relation: Literal["employs", "mitigates", "specializes", "achieves", "sequences"]
    source_id: RequiredText
    target_id: RequiredText
    description: str | None = None
    step_id: str | None = None
    tactic_id: str | None = None
    position: int | None = Field(default=None, ge=0)
    leads_to: list[RequiredText] = Field(default_factory=list)
    source_locator: RequiredText
    source_fields: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("source_locator")
    @classmethod
    def require_locator(cls, value: str) -> str:
        return _require_absolute_locator(value)

    @field_validator("description", "step_id", "tactic_id")
    @classmethod
    def require_nonblank_optional_text(cls, value: str | None) -> str | None:
        return _require_nonblank_optional(value)


class ChunkContext(AppModel):
    """Validated generated retrieval note for one source chunk."""

    context: RequiredText
    support_locators: Annotated[list[RequiredText], Field(min_length=1, max_length=4)]

    @field_validator("context")
    @classmethod
    def require_visible_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("context must contain non-whitespace text")
        return value


class ContextDocument(AppModel):
    """Complete deterministic owner document a chunk is contextualized from."""

    schema_version: VersionOne
    owner_id: RequiredText
    owner_kind: OwnerKind
    document: RequiredText
    support_locators: Annotated[list[RequiredText], Field(min_length=1)]
    document_sha256: Sha256Hex

    @field_validator("support_locators")
    @classmethod
    def require_absolute_locators(cls, values: list[str]) -> list[str]:
        return [_require_absolute_locator(value) for value in values]


class ContextualizationResult(AppModel):
    """One accepted, immutable chunk-context cache entry."""

    schema_version: VersionOne
    cache_key: Sha256Hex
    prompt_version: RequiredText
    model: RequiredText
    response_model: str | None = None
    response_id: str | None = None
    context_text: RequiredText
    support_locators: Annotated[list[RequiredText], Field(min_length=1, max_length=4)]
    embedding_tokens: int = Field(gt=0)
    reranker_tokens: int = Field(gt=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_seconds: float = Field(ge=0)
    request_fingerprint: Sha256Hex
    created_at: RequiredText

    @field_validator("context_text")
    @classmethod
    def require_visible_context(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("context_text must not be blank")
        return value

    @field_validator("response_model", "response_id")
    @classmethod
    def require_nonblank_optional_text(cls, value: str | None) -> str | None:
        return _require_nonblank_optional(value)


class Passage(AppModel):
    """One searchable passage shared by the dense and keyword indexes."""

    schema_version: VersionOne
    passage_id: int = Field(gt=0, lt=2**63)
    generation_id: RequiredText
    entity_id: str | None = None
    edge_id: str | None = None
    owner_kind: OwnerKind
    section: RequiredText
    ordinal: int = Field(ge=0)
    bm25_ordinal: int = Field(ge=0)
    header: RequiredText
    source_text: RequiredText
    context_text: RequiredText
    retrieval_text: RequiredText
    context_cache_key: Sha256Hex
    source_sha256: Sha256Hex
    content_sha256: Sha256Hex
    embedding_tokens: int = Field(gt=0)
    reranker_tokens: int = Field(gt=0)
    source_locator: RequiredText
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)

    @field_validator("source_locator")
    @classmethod
    def require_locator(cls, value: str) -> str:
        return _require_absolute_locator(value)

    @model_validator(mode="after")
    def check_owner_and_offsets(self) -> Self:
        if (self.entity_id is None) == (self.edge_id is None):
            raise ValueError("a passage must have exactly one owner")
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        return self


class Candidate(AppModel):
    """One passage considered by retrieval, with per-channel provenance."""

    passage_id: int = Field(gt=0, lt=2**63)
    channels: Annotated[list[Literal["dense", "bm25"]], Field(min_length=1)]
    dense_rank: int | None = Field(default=None, gt=0)
    dense_distance: float | None = Field(default=None, ge=0)
    bm25_rank: int | None = Field(default=None, gt=0)
    bm25_score: float | None = Field(default=None, ge=0)
    fusion_score: float = Field(ge=0)

    @field_validator("channels")
    @classmethod
    def require_unique_channels(
        cls, values: list[Literal["dense", "bm25"]]
    ) -> list[Literal["dense", "bm25"]]:
        if len(set(values)) != len(values):
            raise ValueError("channels must not repeat")
        return values


class ChunkingIdentity(AppModel):
    """Chunking settings that change passage boundaries."""

    layout_version: RequiredText
    body_target_tokens: int = Field(gt=0)
    body_max_tokens: int = Field(gt=0)
    passage_max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    header_max_tokens: int = Field(gt=0)


class EmbeddingIdentity(AppModel):
    """Embedding model, tokenizer, and artifact identity."""

    model_name: RequiredText
    dimension: int = Field(gt=0)
    query_prefix: RequiredText
    tokenizer: RequiredText
    artifact_sha256: Sha256Hex | None = None


class LexicalIdentity(AppModel):
    """Keyword analyzer and scoring settings recorded with a generation."""

    implementation: Literal["bm25s"]
    analyzer_version: RequiredText
    token_pattern: RequiredText
    lowercase: bool
    stopwords: list[str]
    stemming: bool
    method: Literal["lucene"]
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    dtype: Literal["float32"]
    bm25s_version: str | None = None


class ContextualizationIdentity(AppModel):
    """Non-secret contextualization identity recorded with a generation."""

    document_builder_version: RequiredText
    prompt_version: RequiredText
    model: RequiredText
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"]
    output_tokens: int = Field(gt=0)
    context_max_tokens: int = Field(gt=0)
    temperature: float | None = None
    regeneration_revision: int = Field(ge=0)
    results_digest: Sha256Hex


class GenerationManifest(AppModel):
    """Recorded identity, settings, and counts for one stored generation."""

    schema_version: VersionOne
    generation_id: RequiredText
    created_at: RequiredText
    pipeline_version: RequiredText
    build_fingerprint: Sha256Hex
    snapshot: SourceSnapshot
    entity_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    warnings: list[RequiredText] = Field(default_factory=list)
    passage_count: int = Field(default=0, ge=0)
    context_count: int = Field(default=0, ge=0)
    settings_fingerprint: Sha256Hex | None = None
    artifact_manifest_sha256: Sha256Hex | None = None
    chunking: ChunkingIdentity | None = None
    embedding: EmbeddingIdentity | None = None
    reranker_tokenizer: str | None = None
    lexical: LexicalIdentity | None = None
    contextualization: ContextualizationIdentity | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
