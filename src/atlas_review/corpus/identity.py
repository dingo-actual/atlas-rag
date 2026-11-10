"""Build identity, generation IDs, and the recorded manifest.

The settings fingerprint captures every build-affecting input. The generation
ID derives from it, so a changed setting creates a new generation instead of
rewriting an immutable one.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from typing import cast

from atlas_review.config import AppConfig
from atlas_review.contracts import (
    AtlasEntity,
    AtlasRelationship,
    ChunkingIdentity,
    ContextualizationIdentity,
    EmbeddingIdentity,
    GenerationManifest,
    JsonValue,
    LexicalIdentity,
)
from atlas_review.corpus.atlas_v6 import PIPELINE_VERSION, ParsedCorpus
from atlas_review.corpus.chunk import LAYOUT_VERSION
from atlas_review.corpus.contextualize import contextualization_settings
from atlas_review.hashing import fingerprint, sha256_hex
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.storage.repository import StoredArtifact

RETRIEVAL_PIPELINE_VERSION = "atlas-retrieval-v1"
PACKAGE_NAMES = ("fastembed", "bm25s", "usearch", "tokenizers", "numpy", "openai")


def settings_fingerprint(
    corpus: ParsedCorpus, config: AppConfig, counters: TokenCounters
) -> str:
    """Fingerprint every input that changes the built passages or indexes.

    Args:
        corpus (ParsedCorpus): Parsed source records.
        config (AppConfig): Stored configuration.
        counters (TokenCounters): Tokenizer identities used for the limits.

    Returns:
        (str) SHA-256 digest of the canonical settings identity.

    The fingerprint covers the compatibility-critical inputs the specification
    requires, including the FastEmbed and bm25s package identities.
    """
    identity = cast(
        "JsonValue",
        {
            "source_sha256": corpus.snapshot.source_sha256,
            "pipeline": PIPELINE_VERSION,
            "retrieval_pipeline": RETRIEVAL_PIPELINE_VERSION,
            "layout": LAYOUT_VERSION,
            "chunking": {
                "body_target_tokens": config.chunking.body_target_tokens,
                "body_max_tokens": config.chunking.body_max_tokens,
                "passage_max_tokens": config.chunking.passage_max_tokens,
                "overlap_tokens": config.chunking.overlap_tokens,
                "header_max_tokens": config.chunking.header_max_tokens,
            },
            "contextualization": contextualization_settings(config),
            "tokenizers": [counters.embedding.name, counters.reranker.name],
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
            "vector_index": {
                "metric": config.vector_index.metric,
                "dtype": config.vector_index.dtype,
                "connectivity": config.vector_index.connectivity,
                "expansion_add": config.vector_index.expansion_add,
                "expansion_search": config.vector_index.expansion_search,
            },
            "packages": {
                "fastembed": _package_version("fastembed"),
                "bm25s": _package_version("bm25s"),
            },
        },
    )
    return fingerprint(identity)


def completed_fingerprint(
    settings: str, context_texts: Sequence[str], retrieval_texts: Sequence[str]
) -> str:
    """Fingerprint the completed build, including generated output hashes.

    Args:
        settings (str): Settings fingerprint.
        context_texts (Sequence[str]): Accepted context texts in passage order.
        retrieval_texts (Sequence[str]): Assembled retrieval texts.

    Returns:
        (str) SHA-256 digest of the completed build identity.
    """
    identity: JsonValue = {
        "settings": settings,
        "contexts": [_digest(text) for text in context_texts],
        "passages": [_digest(text) for text in retrieval_texts],
    }
    return fingerprint(identity)


def generation_id(settings: str) -> str:
    """Derive the immutable generation identifier from the settings.

    Args:
        settings (str): Settings fingerprint.

    Returns:
        (str) UUID string used as the generation identifier.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, settings))


def restamp(corpus: ParsedCorpus, identifier: str) -> ParsedCorpus:
    """Return the corpus with every record bound to the build generation.

    Args:
        corpus (ParsedCorpus): Parsed source records.
        identifier (str): Build generation identifier.

    Returns:
        (ParsedCorpus) Corpus whose records carry the build generation ID.
    """
    entities = tuple(
        AtlasEntity.model_validate(
            {**entity.model_dump(mode="python"), "generation_id": identifier},
            strict=True,
        )
        for entity in corpus.entities
    )
    edges = tuple(
        AtlasRelationship.model_validate(
            {**edge.model_dump(mode="python"), "generation_id": identifier},
            strict=True,
        )
        for edge in corpus.edges
    )
    return replace(corpus, generation_id=identifier, entities=entities, edges=edges)


@dataclass(frozen=True, slots=True)
class BuildRecord:
    """Completed build facts recorded in a generation manifest.

    Instance Attributes:
        settings (str): Settings fingerprint.
        build (str): Completed build fingerprint.
        context_texts (tuple[str, ...]): Accepted context texts in passage order.
        passage_count (int): Stored passage count.
        created_at (str): ISO 8601 generation timestamp.
    """

    settings: str
    build: str
    context_texts: tuple[str, ...]
    passage_count: int
    created_at: str


def manifest(
    *,
    corpus: ParsedCorpus,
    config: AppConfig,
    counters: TokenCounters,
    record: BuildRecord,
) -> GenerationManifest:
    """Build the recorded manifest for one completed generation.

    Args:
        corpus (ParsedCorpus): Generation records.
        config (AppConfig): Configuration snapshot of the build.
        counters (TokenCounters): Tokenizer identities.
        record (BuildRecord): Completed build facts.

    Returns:
        (GenerationManifest) Provisional manifest without bundle hashes.
    """
    context_texts = record.context_texts
    return GenerationManifest(
        schema_version=1,
        generation_id=corpus.generation_id,
        created_at=record.created_at,
        pipeline_version=RETRIEVAL_PIPELINE_VERSION,
        build_fingerprint=record.build,
        snapshot=corpus.snapshot,
        entity_count=len(corpus.entities),
        edge_count=len(corpus.edges),
        warnings=list(corpus.warnings),
        passage_count=record.passage_count,
        context_count=len(context_texts),
        settings_fingerprint=record.settings,
        chunking=ChunkingIdentity(
            layout_version=LAYOUT_VERSION,
            body_target_tokens=config.chunking.body_target_tokens,
            body_max_tokens=config.chunking.body_max_tokens,
            passage_max_tokens=config.chunking.passage_max_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
            header_max_tokens=config.chunking.header_max_tokens,
        ),
        embedding=EmbeddingIdentity(
            model_name=config.embedding.model_name,
            dimension=config.embedding.dimension,
            query_prefix=config.embedding.query_prefix,
            tokenizer=counters.embedding.name,
        ),
        reranker_tokenizer=counters.reranker.name,
        lexical=LexicalIdentity(
            implementation=config.keyword_index.implementation,
            analyzer_version=config.keyword_index.analyzer_version,
            token_pattern=config.keyword_index.token_pattern,
            lowercase=config.keyword_index.lowercase,
            stopwords=list(config.keyword_index.stopwords),
            stemming=config.keyword_index.stemming,
            method=config.keyword_index.method,
            k1=config.keyword_index.k1,
            b=config.keyword_index.b,
            dtype=config.keyword_index.dtype,
            bm25s_version=_package_version("bm25s"),
        ),
        contextualization=ContextualizationIdentity(
            document_builder_version=config.contextualization.document_builder_version,
            prompt_version=config.contextualization.prompt_version,
            model=config.llm.model,
            output_token_parameter=config.llm.output_token_parameter,
            output_tokens=config.contextualization.output_tokens,
            context_max_tokens=config.contextualization.context_max_tokens,
            temperature=config.contextualization.temperature,
            reasoning_effort=config.llm.reasoning_effort,
            regeneration_revision=config.contextualization.regeneration_revision,
            results_digest=fingerprint([_digest(text) for text in context_texts]),
        ),
        package_versions=package_versions(),
    )


def with_artifact(
    manifest_value: GenerationManifest, artifact: StoredArtifact
) -> GenerationManifest:
    """Return the manifest with its published bundle hash recorded."""
    return GenerationManifest.model_validate(
        {
            **manifest_value.model_dump(mode="python"),
            "artifact_manifest_sha256": artifact.manifest_sha256,
        },
        strict=True,
    )


def package_versions() -> dict[str, str]:
    """Return compat-critical package versions for diagnosis."""
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        version = _package_version(name)
        if version is not None:
            versions[name] = version
    return versions


def timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()


def _package_version(name: str) -> str | None:
    """Return an installed package version, or ``None`` when unavailable."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _digest(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return sha256_hex(text.encode("utf-8"))
