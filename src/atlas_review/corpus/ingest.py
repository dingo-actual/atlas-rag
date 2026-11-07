"""Explicit contextualized ingestion: plan, contextualize, index, and activate.

Planning is LLM-free. Only uncached chunks require the configured provider.
Both derived indexes publish as one immutable bundle, activated in the same
transaction that records the generation, so a failure leaves the previous
generation active.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from atlas_review.config import AppConfig
from atlas_review.contracts import (
    ChunkContext,
    ContextDocument,
    ContextualizationResult,
    Passage,
)
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.chunk import SourceChunk, build_source_chunks
from atlas_review.corpus.contextualize import (
    SYSTEM_PROMPT,
    AcceptedContext,
    assemble_retrieval_text,
    build_owner_documents,
    build_user_payload,
    context_cache_key,
    parse_chunk_context,
    validate_context,
)
from atlas_review.corpus.identity import (
    BuildRecord,
    completed_fingerprint,
    generation_id,
    restamp,
    settings_fingerprint,
    timestamp,
    with_artifact,
)
from atlas_review.corpus.identity import (
    manifest as build_manifest,
)
from atlas_review.errors import (
    STAGE_CONTEXT_DOCUMENT,
    STAGE_INGESTION,
    AtlasReviewError,
)
from atlas_review.hashing import sha256_hex
from atlas_review.inference.client import Contextualizer, ProviderReply
from atlas_review.retrieval.artifacts import Publication, build_bundle
from atlas_review.retrieval.embeddings import EmbeddingBackend, passage_vectors
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.storage.passages import PassageRepository
from atlas_review.storage.repository import CorpusRepository, StoredArtifact

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    """What one explicit build would do, before any provider request.

    Instance Attributes:
        generation_id (str): Generation identity derived from the settings.
        settings_fingerprint (str): Fingerprint of the build-affecting settings.
        chunk_count (int): Planned source chunks.
        cached_context_count (int): Chunks with a validated cached context.
        uncached_context_count (int): Chunks that need a provider request.
        reuse_generation (bool): An identical ready generation already exists.
        endpoint_host (str | None): Host that would be called for missing context.
        model (str): Configured model identifier.
        context_output_tokens (int): Output allowance per context request.
    """

    generation_id: str
    settings_fingerprint: str
    chunk_count: int
    cached_context_count: int
    uncached_context_count: int
    reuse_generation: bool
    endpoint_host: str | None
    model: str
    context_output_tokens: int


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    """Result of one explicit contextualized ingestion.

    Instance Attributes:
        generation_id (str): Active generation after the operation.
        passage_count (int): Stored searchable passages.
        context_count (int): Accepted context entries used by the generation.
        cached_context_count (int): Context entries reused from the cache.
        requested_context_count (int): Provider requests made by this run.
        dense_count (int): Vectors in the published dense index.
        keyword_count (int): Documents in the published keyword index.
        reused_generation (bool): The identical generation was already ready.
        artifact_directory (str): Published bundle path inside the workspace.
    """

    generation_id: str
    passage_count: int
    context_count: int
    cached_context_count: int
    requested_context_count: int
    dense_count: int
    keyword_count: int
    reused_generation: bool
    artifact_directory: str


@dataclass(frozen=True, slots=True)
class PreparedBuild:
    """Derived inputs shared by planning and building.

    Instance Attributes:
        corpus (ParsedCorpus): Records re-stamped with the build generation ID.
        documents (dict[str, ContextDocument]): Owner documents.
        chunks (tuple[SourceChunk, ...]): Searchable chunks in stable order.
        cache_keys (tuple[str, ...]): Cache key for each chunk.
        settings_fingerprint (str): Fingerprint of the build-affecting settings.
    """

    corpus: ParsedCorpus
    documents: dict[str, ContextDocument]
    chunks: tuple[SourceChunk, ...]
    cache_keys: tuple[str, ...]
    settings_fingerprint: str


def prepare_build(
    corpus: ParsedCorpus, *, config: AppConfig, counters: TokenCounters
) -> PreparedBuild:
    """Build owner documents, chunks, and cache keys deterministically.

    Args:
        corpus (ParsedCorpus): Parsed source records.
        config (AppConfig): Stored configuration.
        counters (TokenCounters): Retrieval tokenizers.

    Returns:
        (PreparedBuild) Derived inputs for planning and building.
    """
    settings = settings_fingerprint(corpus, config, counters)
    restamped = restamp(corpus, generation_id(settings))
    documents = build_owner_documents(
        restamped.entities, restamped.edges, config.contextualization
    )
    chunks = build_source_chunks(
        restamped.entities, restamped.edges, counters.embedding, config.chunking
    )
    cache_keys = tuple(
        context_cache_key(
            document=documents[chunk.owner_id],
            chunk=chunk,
            config=config,
            counters=counters,
        )
        for chunk in chunks
    )
    return PreparedBuild(
        corpus=restamped,
        documents=documents,
        chunks=chunks,
        cache_keys=cache_keys,
        settings_fingerprint=settings,
    )


def plan_ingestion(
    corpus: ParsedCorpus,
    *,
    config: AppConfig,
    repository: CorpusRepository,
    passages: PassageRepository,
    counters: TokenCounters,
) -> IngestionPlan:
    """Describe one explicit build without calling any model.

    Args:
        corpus (ParsedCorpus): Parsed source records.
        config (AppConfig): Stored configuration.
        repository (CorpusRepository): Generation storage.
        passages (PassageRepository): Context-cache storage.
        counters (TokenCounters): Retrieval tokenizers.

    Returns:
        (IngestionPlan) Chunk counts, cache reuse, and the target endpoint.
    """
    prepared = prepare_build(corpus, config=config, counters=counters)
    return plan_from_build(
        prepared, config=config, repository=repository, passages=passages
    )


def plan_from_build(
    prepared: PreparedBuild,
    *,
    config: AppConfig,
    repository: CorpusRepository,
    passages: PassageRepository,
) -> IngestionPlan:
    """Describe one explicit build from already-prepared inputs.

    Args:
        prepared (PreparedBuild): Derived documents, chunks, and cache keys.
        config (AppConfig): Stored configuration.
        repository (CorpusRepository): Generation storage.
        passages (PassageRepository): Context-cache storage.

    Returns:
        (IngestionPlan) Chunk counts, cache reuse, and the target endpoint.
    """
    identifier = generation_id(prepared.settings_fingerprint)
    record = repository.generation(identifier)
    reuse = record is not None and record.status == "ready"
    cached = (
        len(prepared.chunks)
        if reuse
        else sum(
            1 for key in prepared.cache_keys if passages.context_entry(key) is not None
        )
    )
    return IngestionPlan(
        generation_id=identifier,
        settings_fingerprint=prepared.settings_fingerprint,
        chunk_count=len(prepared.chunks),
        cached_context_count=cached,
        uncached_context_count=len(prepared.chunks) - cached,
        reuse_generation=reuse,
        endpoint_host=urlsplit(config.llm.base_url).hostname,
        model=config.llm.model,
        context_output_tokens=config.contextualization.output_tokens,
    )


@dataclass(frozen=True, slots=True)
class BuildServices:
    """Collaborators an explicit build needs.

    Instance Attributes:
        config (AppConfig): Stored configuration.
        repository (CorpusRepository): Generation and activation storage.
        passages (PassageRepository): Passage and context-cache storage.
        indexes_dir (Path): Workspace index directory.
        counters (TokenCounters): Retrieval tokenizers.
        backend (EmbeddingBackend): Embedding implementation.
        contextualizer (Contextualizer | None): Provider client for uncached chunks.
        progress (ProgressCallback | None): Stage progress reporter.
    """

    config: AppConfig
    repository: CorpusRepository
    passages: PassageRepository
    indexes_dir: Path
    counters: TokenCounters
    backend: EmbeddingBackend
    contextualizer: Contextualizer | None
    progress: ProgressCallback | None


def run_ingestion(
    corpus: ParsedCorpus, *, source_bytes: bytes, services: BuildServices
) -> IngestionOutcome:
    """Contextualize and index one source, then activate it atomically.

    Args:
        corpus (ParsedCorpus): Parsed source records.
        source_bytes (bytes): Exact source bytes retained for provenance.
        services (BuildServices): Collaborators for the build.

    Returns:
        (IngestionOutcome) Activation summary.

    Raises:
        AtlasReviewError: Contextualization, embedding, or index publication
            failed; the previously active generation stays active.
    """
    prepared = prepare_build(corpus, config=services.config, counters=services.counters)
    plan = plan_from_build(
        prepared,
        config=services.config,
        repository=services.repository,
        passages=services.passages,
    )
    progress = services.progress if services.progress is not None else _ignore
    if plan.reuse_generation:
        services.repository.activate(plan.generation_id)
        return _reused_outcome(services.repository, plan.generation_id)
    contexts, cached_count, requested_count = _resolve_contexts(prepared, services)
    progress("assembling augmented retrieval text")
    if not prepared.chunks:
        raise _no_passages_error()
    retrieval_texts = tuple(
        assemble_retrieval_text(chunk.header, contexts[key].context_text, chunk.text)
        for chunk, key in zip(prepared.chunks, prepared.cache_keys, strict=True)
    )
    progress(f"embedding {len(retrieval_texts)} passages")
    vectors = passage_vectors(
        services.backend, services.config.embedding, retrieval_texts
    )
    context_texts = tuple(contexts[key].context_text for key in prepared.cache_keys)
    build = completed_fingerprint(
        prepared.settings_fingerprint, context_texts, retrieval_texts
    )
    progress("publishing the dense and keyword index bundle")
    with services.repository.transaction() as connection:
        base_id = services.passages.next_passage_id(connection)
        stored_passages = _passages(
            prepared, contexts, retrieval_texts=retrieval_texts, base_id=base_id
        )
        manifest = build_manifest(
            corpus=prepared.corpus,
            config=services.config,
            counters=services.counters,
            record=BuildRecord(
                settings=prepared.settings_fingerprint,
                build=build,
                context_texts=context_texts,
                passage_count=len(stored_passages),
                created_at=timestamp(),
            ),
        )
        services.repository.insert_generation(
            connection, prepared.corpus, manifest, source_bytes
        )
        services.passages.insert_passages(
            connection, stored_passages, [_vector_blob(row) for row in vectors]
        )
        bundle = build_bundle(
            publication=Publication(
                indexes_dir=services.indexes_dir,
                generation_id=plan.generation_id,
                artifact_id=build[:16],
            ),
            vectors=vectors,
            passage_ids=[passage.passage_id for passage in stored_passages],
            retrieval_texts=retrieval_texts,
            config=services.config,
        )
        artifact = StoredArtifact(
            directory=bundle.directory.relative_to(
                services.indexes_dir.parent
            ).as_posix(),
            manifest_sha256=bundle.manifest_sha256,
        )
        services.repository.finalize_generation(
            connection,
            manifest=with_artifact(manifest, artifact),
            artifact=artifact,
        )
    progress(f"activated generation {plan.generation_id}")
    return IngestionOutcome(
        generation_id=plan.generation_id,
        passage_count=len(stored_passages),
        context_count=len(prepared.cache_keys),
        cached_context_count=cached_count,
        requested_context_count=requested_count,
        dense_count=len(stored_passages),
        keyword_count=len(stored_passages),
        reused_generation=False,
        artifact_directory=artifact.directory,
    )


def _resolve_contexts(
    prepared: PreparedBuild, services: BuildServices
) -> tuple[dict[str, AcceptedContext], int, int]:
    """Reuse validated cached context and request only the missing chunks."""
    config = services.config
    counters = services.counters
    report = services.progress if services.progress is not None else _ignore
    accepted: dict[str, AcceptedContext] = {}
    cached_count = 0
    requested_count = 0
    total = len(prepared.chunks)
    for position, (chunk, key) in enumerate(
        zip(prepared.chunks, prepared.cache_keys, strict=True), start=1
    ):
        document = prepared.documents[chunk.owner_id]
        entry = services.passages.context_entry(key)
        if entry is not None:
            accepted[key] = validate_context(
                ChunkContext(
                    context=entry.context_text,
                    support_locators=entry.support_locators,
                ),
                document=document,
                chunk=chunk,
                config=config,
                counters=counters,
            )
            cached_count += 1
            continue
        contextualizer = services.contextualizer
        if contextualizer is None:
            raise AtlasReviewError(
                code="contextualizer_required",
                stage=STAGE_INGESTION,
                message=(
                    f"chunk {position}/{total} of {document.owner_id} has no cached "
                    "context and no provider client is available"
                ),
                recovery=(
                    "Configure the provider and its credential variable, then "
                    "resume the explicit build."
                ),
            )
        payload = _request_payload(document, chunk, config)
        report(f"contextualizing {position}/{total} ({document.owner_id})")
        reply = contextualizer(
            SYSTEM_PROMPT,
            payload,
            output_tokens=config.contextualization.output_tokens,
            temperature=config.contextualization.temperature,
        )
        validated = validate_context(
            parse_chunk_context(reply.text),
            document=document,
            chunk=chunk,
            config=config,
            counters=counters,
        )
        services.passages.save_context_entry(
            _context_result(validated, key=key, reply=reply, config=config),
            document_sha256=document.document_sha256,
            chunk_sha256=sha256_hex(chunk.text.encode("utf-8")),
        )
        accepted[key] = validated
        requested_count += 1
    return accepted, cached_count, requested_count


def _context_result(
    accepted: AcceptedContext, *, key: str, reply: ProviderReply, config: AppConfig
) -> ContextualizationResult:
    """Build the immutable cache record for one accepted response."""
    return ContextualizationResult(
        schema_version=1,
        cache_key=key,
        prompt_version=config.contextualization.prompt_version,
        model=config.llm.model,
        response_model=reply.model,
        response_id=reply.response_id,
        context_text=accepted.context_text,
        support_locators=list(accepted.support_locators),
        embedding_tokens=accepted.embedding_tokens,
        reranker_tokens=accepted.reranker_tokens,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        duration_seconds=reply.duration_seconds,
        request_fingerprint=key,
        created_at=timestamp(),
    )


def _request_payload(
    document: ContextDocument, chunk: SourceChunk, config: AppConfig
) -> str:
    """Build the request payload and enforce the character budget."""
    payload = build_user_payload(document, chunk)
    size = len(SYSTEM_PROMPT) + len(payload)
    limit = config.contextualization.max_input_characters
    if size > limit:
        raise AtlasReviewError(
            code="context_document_too_large",
            stage=STAGE_CONTEXT_DOCUMENT,
            message=(
                f"the owner document for {document.owner_id} needs {size} "
                f"characters; the limit is {limit}"
            ),
            recovery="Reduce the document size or adjust the request limit explicitly.",
        )
    return payload


def _passages(
    prepared: PreparedBuild,
    contexts: Mapping[str, AcceptedContext],
    *,
    retrieval_texts: Sequence[str],
    base_id: int,
) -> tuple[Passage, ...]:
    """Build validated passage records in stable BM25 ordinal order."""
    passages: list[Passage] = []
    triples = zip(prepared.chunks, prepared.cache_keys, retrieval_texts, strict=True)
    for position, (chunk, key, retrieval_text) in enumerate(triples):
        accepted = contexts[key]
        passages.append(
            Passage(
                schema_version=1,
                passage_id=base_id + position,
                generation_id=prepared.corpus.generation_id,
                entity_id=chunk.entity_id,
                edge_id=chunk.edge_id,
                owner_kind=chunk.owner_kind,
                section=chunk.section,
                ordinal=chunk.ordinal,
                bm25_ordinal=position,
                header=chunk.header,
                source_text=chunk.text,
                context_text=accepted.context_text,
                retrieval_text=retrieval_text,
                context_cache_key=key,
                source_sha256=sha256_hex(chunk.text.encode("utf-8")),
                content_sha256=sha256_hex(retrieval_text.encode("utf-8")),
                embedding_tokens=accepted.passage_embedding_tokens,
                reranker_tokens=accepted.passage_reranker_tokens,
                source_locator=chunk.source_locator,
                source_start=chunk.start,
                source_end=chunk.end,
            )
        )
    return tuple(passages)


def _vector_blob(row: np.ndarray) -> bytes:
    """Encode one normalized vector as an explicit little-endian float32 blob."""
    return np.asarray(row, dtype="<f4").tobytes(order="C")


def _reused_outcome(repository: CorpusRepository, identifier: str) -> IngestionOutcome:
    """Report an already ready generation that was reactivated."""
    stored = repository.manifest(identifier)
    artifact = repository.artifact(identifier)
    return IngestionOutcome(
        generation_id=identifier,
        passage_count=stored.passage_count,
        context_count=stored.context_count,
        cached_context_count=stored.context_count,
        requested_context_count=0,
        dense_count=stored.passage_count,
        keyword_count=stored.passage_count,
        reused_generation=True,
        artifact_directory=artifact.directory if artifact is not None else "",
    )


def _ignore(message: str) -> None:
    """Discard a progress message when no reporter is supplied."""
    del message


def _no_passages_error() -> AtlasReviewError:
    """Build the error raised when a source produced no searchable passages."""
    return AtlasReviewError(
        code="empty_generation",
        stage=STAGE_INGESTION,
        message="the source produced no searchable passages",
        recovery="Select an ATLAS source whose records have searchable descriptions.",
    )
