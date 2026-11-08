"""Workspace session: one configured workspace used by the terminal UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

from atlas_review.config import AppConfig
from atlas_review.contracts import (
    ContextualizationResult,
    EvidencePacket,
    Passage,
    ReviewAttempt,
    ReviewRequest,
)
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.ingest import (
    BuildServices,
    IngestionPlan,
    plan_ingestion,
    run_ingestion,
)
from atlas_review.corpus.ingest import IngestionOutcome as BuildOutcome
from atlas_review.errors import STAGE_INGESTION, STAGE_RETRIEVAL, AtlasReviewError
from atlas_review.export import export_markdown
from atlas_review.inference.client import (
    ChatContextualizer,
    build_contextualizer,
    build_reviewer,
)
from atlas_review.inference.prompt import ReviewPrompt, build_prompt
from atlas_review.inference.review import (
    AttemptPlan,
    Reviewer,
    ReviewOutcome,
    generate_review,
)
from atlas_review.retrieval.embeddings import EmbeddingBackend, FastEmbedBackend
from atlas_review.retrieval.hybrid import SearchOutcome, check_focus, search
from atlas_review.retrieval.loader import load_generation
from atlas_review.retrieval.pipeline import RetrievalRun, retrieve
from atlas_review.retrieval.rerank import FastEmbedReranker, Reranker
from atlas_review.retrieval.tokenizers import TokenCounters, load_token_counters
from atlas_review.services.ingestion import (
    IngestionOutcome,
    SourcePreview,
    ingest_source,
    preview_source,
    read_source_file,
)
from atlas_review.services.settings import load_config, save_config
from atlas_review.storage.database import WorkspaceDatabase
from atlas_review.storage.passages import PassageRepository
from atlas_review.storage.repository import (
    CorpusRepository,
    EntityInspection,
    GenerationSummary,
)
from atlas_review.storage.reviews import (
    ReviewRepository,
    ReviewRetrieval,
    ReviewSummary,
    ReviewWriter,
    StoredReview,
)


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Filesystem layout of one workspace directory.

    Instance Attributes:
        root (Path): Workspace directory supplied by ``--workspace``.
        database (Path): SQLite database file.
        indexes (Path): Published index bundles.
        models (Path): Downloaded model cache.
        exports (Path): Exported reports.
    """

    root: Path
    database: Path
    indexes: Path
    models: Path
    exports: Path

    @classmethod
    def under(cls, root: Path) -> WorkspacePaths:
        """Derive the documented workspace layout from a root directory.

        Args:
            root (Path): Workspace directory.

        Returns:
            (WorkspacePaths) Paths inside the root directory.
        """
        return cls(
            root=root,
            database=root / "atlas.sqlite3",
            indexes=root / "indexes",
            models=root / "models",
            exports=root / "exports",
        )


@dataclass(frozen=True, slots=True)
class PreparedIngestion:
    """A planned source ready for an explicit contextualized build.

    Instance Attributes:
        preview (SourcePreview): Parsed source summary for confirmation.
        plan (IngestionPlan): LLM-free build plan.
        corpus (ParsedCorpus): Parsed records retained from preparation.
        source_bytes (bytes): Exact source bytes retained for provenance.
    """

    preview: SourcePreview
    plan: IngestionPlan
    corpus: ParsedCorpus
    source_bytes: bytes


@dataclass(frozen=True, slots=True)
class ReviewDraft:
    """One retrieved review waiting for an explicit Generate action.

    Instance Attributes:
        review_id (str): Stored review identifier.
        brief (ReviewRequest): Validated brief used for retrieval.
        run (RetrievalRun): Local retrieval result whose packet was frozen.
    """

    review_id: str
    brief: ReviewRequest
    run: RetrievalRun


@dataclass(frozen=True, slots=True)
class GeneratedContext:
    """One stored passage with the cached note generated for it.

    Instance Attributes:
        passage (Passage): Stored passage with its source and retrieval text.
        entry (ContextualizationResult | None): Accepted cache entry behind the
            note, or ``None`` when the cache row is unavailable.
    """

    passage: Passage
    entry: ContextualizationResult | None


class WorkspaceSession:
    """Coordinate one workspace database, configuration, and corpus repository."""

    def __init__(
        self,
        root: Path,
        *,
        counters: TokenCounters | None = None,
        embedding: EmbeddingBackend | None = None,
        reranker: Reranker | None = None,
        reviewer: Reviewer | None = None,
    ) -> None:
        """Open the workspace and load or initialise its configuration.

        Args:
            root (Path): Workspace directory supplied by ``--workspace``.
            counters (TokenCounters | None): Injected retrieval tokenizers.
            embedding (EmbeddingBackend | None): Injected embedding backend.
            reranker (Reranker | None): Injected cross-encoder backend.
            reviewer (Reviewer | None): Injected review-synthesis adapter.
        """
        self.paths = WorkspacePaths.under(root)
        for directory in (self.paths.indexes, self.paths.models, self.paths.exports):
            directory.mkdir(parents=True, exist_ok=True)
        self.database = WorkspaceDatabase(self.paths.database)
        self.repository = CorpusRepository(self.database)
        self.passages = PassageRepository(self.database)
        self.reviews = ReviewRepository(self.database)
        self.reviews.interrupt_started()
        self.config: AppConfig = load_config(self.database, updated_at=_timestamp())
        self._counters = counters
        self._embedding = embedding
        self._reranker = reranker
        self._reviewer = reviewer
        self._review_client: ChatContextualizer | None = None

    def close(self) -> None:
        """Close the owned review client and the workspace database connection."""
        self.close_review_client()
        self.database.close()

    def close_review_client(self) -> None:
        """Close the cached review client so a new configuration starts fresh."""
        if self._review_client is not None:
            self._review_client.close()
            self._review_client = None

    def review_client(self) -> Reviewer:
        """Build the review adapter once per configuration, then reuse it."""
        if self._reviewer is not None:
            return self._reviewer
        if self._review_client is None:
            self._review_client = build_reviewer(self.config.llm)
        return self._review_client

    def token_counters(self) -> TokenCounters:
        """Load the retrieval tokenizers once per session."""
        if self._counters is None:
            self._counters = load_token_counters(self.config, self.paths.models)
        return self._counters

    def embedding_backend(self) -> EmbeddingBackend:
        """Load the embedding session once per session."""
        if self._embedding is None:
            self._embedding = FastEmbedBackend(self.config.embedding, self.paths.models)
        return self._embedding

    def reranker_backend(self) -> Reranker:
        """Load the cross-encoder session once per session."""
        if self._reranker is None:
            self._reranker = FastEmbedReranker(self.config.reranker, self.paths.models)
        return self._reranker

    def prepare_ingestion(self, path: Path) -> PreparedIngestion:
        """Read a source, plan its build, and keep the bytes for later.

        Args:
            path (Path): ATLAS v6 YAML file.

        Returns:
            (PreparedIngestion) Corpus, preview, and LLM-free build plan.
        """
        raw = read_source_file(path, max_bytes=self.config.corpus.max_source_bytes)
        corpus, preview = preview_source(raw, origin=str(path))
        plan = plan_ingestion(
            corpus,
            config=self.config,
            repository=self.repository,
            passages=self.passages,
            counters=self.token_counters(),
        )
        return PreparedIngestion(
            preview=preview, plan=plan, corpus=corpus, source_bytes=raw
        )

    def build_generation(
        self, prepared: PreparedIngestion, progress: Callable[[str], None] | None = None
    ) -> BuildOutcome:
        """Contextualize and index a planned source, then activate it.

        Args:
            prepared (PreparedIngestion): Planned source from preparation.
            progress (Callable[[str], None] | None): Stage progress reporter.

        Returns:
            (IngestionOutcome) Activation summary.
        """
        contextualizer = None
        if not prepared.plan.reuse_generation and prepared.plan.uncached_context_count:
            contextualizer = build_contextualizer(self.config.llm)
        return run_ingestion(
            prepared.corpus,
            source_bytes=prepared.source_bytes,
            services=BuildServices(
                config=self.config,
                repository=self.repository,
                passages=self.passages,
                indexes_dir=self.paths.indexes,
                counters=self.token_counters(),
                backend=self.embedding_backend(),
                contextualizer=contextualizer,
                progress=progress,
            ),
        )

    def search(self, focus: str) -> SearchOutcome:
        """Run the initial dense and keyword channels over the active generation.

        Args:
            focus (str): Ordinary retrieval focus text.

        Returns:
            (SearchOutcome) Channel hits, fused candidates, and passages.
        """
        generation_id = self.require_ready_generation_id()
        counters = self.token_counters()
        check_focus(focus, counters, self.config)
        loaded = load_generation(
            self.repository,
            passages=self.passages,
            indexes_dir=self.paths.indexes,
            config=self.config,
            generation_id=generation_id,
        )
        return search(loaded, self.embedding_backend(), self.config, focus)

    def retrieve(self, focus: str) -> RetrievalRun:
        """Run the full local retrieval pipeline over the active generation.

        Args:
            focus (str): Ordinary retrieval focus text.

        Returns:
            (RetrievalRun) Traceable retrieval result with a frozen packet.
        """
        generation_id = self.require_ready_generation_id()
        loaded = load_generation(
            self.repository,
            passages=self.passages,
            indexes_dir=self.paths.indexes,
            config=self.config,
            generation_id=generation_id,
        )
        starting = monotonic()
        counters = self.token_counters()
        backend = self.embedding_backend()
        reranker = self.reranker_backend()
        model_load_seconds = monotonic() - starting
        return retrieve(
            loaded,
            backend,
            reranker,
            self.config,
            focus,
            counters,
            model_load_seconds=model_load_seconds,
        )

    def ingest_file(self, path: Path) -> IngestionOutcome:
        """Read, parse, and store one local source file.

        Args:
            path (Path): ATLAS v6 YAML file.

        Returns:
            (IngestionOutcome) Stored generation identifier and preview.
        """
        raw = read_source_file(path, max_bytes=self.config.corpus.max_source_bytes)
        return ingest_source(self.repository, raw, origin=str(path))

    def preview_file(self, path: Path) -> SourcePreview:
        """Parse one local source file and summarize it without storing it.

        Args:
            path (Path): ATLAS v6 YAML file.

        Returns:
            (SourcePreview) Summary for the confirmation prompt.
        """
        raw = read_source_file(path, max_bytes=self.config.corpus.max_source_bytes)
        _, preview = preview_source(raw, origin=str(path))
        return preview

    def generations(self) -> tuple[GenerationSummary, ...]:
        """Return stored generations newest first."""
        return self.repository.summaries()

    def save_config(self, config: AppConfig) -> None:
        """Validate and store a replacement configuration.

        Args:
            config (AppConfig): Complete replacement configuration.
        """
        validated = AppConfig.model_validate(config, strict=True)
        save_config(self.database, validated, updated_at=_timestamp())
        self.config = validated
        self.close_review_client()

    def retrieve_review(self, brief: ReviewRequest) -> ReviewDraft:
        """Retrieve local evidence for a brief and freeze it as a saved review.

        Args:
            brief (ReviewRequest): Validated application brief.

        Returns:
            (ReviewDraft) Stored review identifier and the retrieval run.
        """
        run = self.retrieve(brief.retrieval_focus)
        review_id = uuid4().hex
        self.reviews.save_retrieved(
            ReviewRetrieval(
                review_id=review_id,
                generation_id=run.trace.generation_id,
                created_at=_timestamp(),
                request=brief,
                config=self.config,
                trace=run.trace,
                evidence=run.selection.packet,
            )
        )
        return ReviewDraft(review_id=review_id, brief=brief, run=run)

    def generate_report(self, draft: ReviewDraft) -> ReviewOutcome:
        """Make one explicit review request for a retrieved draft.

        Args:
            draft (ReviewDraft): Draft whose frozen packet is synthesized.

        Returns:
            (ReviewOutcome) Finished attempt and validated report.
        """
        return self._synthesize(
            draft.review_id, draft.brief, draft.run.selection.packet
        )

    def prepare_review_request(self, draft: ReviewDraft) -> ReviewPrompt:
        """Build the synthesis request the Generate action will send.

        The request applies the configured character budget, so the screen can
        show the evidence that actually fits before any provider call.

        Args:
            draft (ReviewDraft): Draft holding the frozen evidence packet.

        Returns:
            (ReviewPrompt) System message, user payload, and trimmed packet.
        """
        return build_prompt(draft.brief, draft.run.selection.packet, self.config.llm)

    def regenerate_review(self, review_id: str) -> ReviewOutcome:
        """Make one new attempt for a reopened review.

        The brief and evidence are the frozen ones stored with the review; the
        currently configured provider settings are used and recorded.

        Args:
            review_id (str): Stored review identifier.

        Returns:
            (ReviewOutcome) Finished attempt and validated report.
        """
        stored = self.reviews.review(review_id)
        return self._synthesize(review_id, stored.request, stored.evidence)

    def _synthesize(
        self, review_id: str, brief: ReviewRequest, packet: EvidencePacket
    ) -> ReviewOutcome:
        """Record one attempt around exactly one provider request."""
        return generate_review(
            brief,
            packet,
            self.config.llm,
            plan=AttemptPlan(
                review_id=review_id,
                attempt_number=self.reviews.next_attempt_number(review_id),
                started_at=_timestamp(),
            ),
            reviewer=self.review_client(),
            store=ReviewWriter(self.reviews, review_id),
        )

    def saved_reviews(self) -> tuple[ReviewSummary, ...]:
        """Return saved reviews newest first."""
        return self.reviews.summaries()

    def open_review(self, review_id: str) -> StoredReview:
        """Reopen one saved review without any provider request.

        Args:
            review_id (str): Stored review identifier.

        Returns:
            (StoredReview) Saved brief, evidence, and optional report.
        """
        return self.reviews.review(review_id)

    def review_attempts(self, review_id: str) -> tuple[ReviewAttempt, ...]:
        """Return every recorded attempt of one saved review."""
        return self.reviews.attempts(review_id)

    def generated_context(
        self, generation_id: str, passage_id: int
    ) -> GeneratedContext:
        """Return one stored passage with its cached generated note.

        Args:
            generation_id (str): Generation the passage belongs to.
            passage_id (int): Stored passage identifier.

        Returns:
            (GeneratedContext) Passage and its accepted cache entry, if any.
        """
        passage = self.passages.passage(generation_id, passage_id)
        return GeneratedContext(
            passage=passage,
            entry=self.passages.context_entry(passage.context_cache_key),
        )

    def export_review(self, review_id: str, *, overwrite: bool = False) -> Path:
        """Export one saved review as Markdown into the workspace exports.

        Args:
            review_id (str): Stored review identifier.
            overwrite (bool): Replace an existing export file.

        Returns:
            (Path) Written Markdown file.
        """
        return export_markdown(
            self.reviews.review(review_id), self.paths.exports, overwrite=overwrite
        )

    def current_generation_id(self) -> str | None:
        """Return the active generation, or the newest stored one."""
        return (
            self.repository.active_generation_id()
            or self.repository.latest_generation_id()
        )

    def entity_choices(self, kind: str) -> tuple[tuple[str, str], ...]:
        """Return ``(entity_id, display_name)`` pairs for one kind.

        Args:
            kind (str): Entity kind such as ``technique`` or ``mitigation``.

        Returns:
            (tuple[tuple[str, str], ...]) Selectable entities.

        Raises:
            AtlasReviewError: No generation is stored yet.
        """
        generation_id = self.require_generation_id()
        return self.repository.entity_ids(generation_id, kind)

    def inspect(self, entity_id: str) -> EntityInspection:
        """Return one entity with both relationship directions.

        Args:
            entity_id (str): Stored entity identifier.

        Returns:
            (EntityInspection) Entity and its provenance edges.
        """
        return self.repository.inspect(self.require_generation_id(), entity_id)

    def require_generation_id(self) -> str:
        """Return the current generation identifier or fail with guidance.

        Returns:
            (str) Active or newest generation identifier.

        Raises:
            AtlasReviewError: The workspace stores no generation yet.
        """
        generation_id = self.current_generation_id()
        if generation_id is None:
            raise AtlasReviewError(
                code="no_generation",
                stage=STAGE_INGESTION,
                message="this workspace has no stored corpus generation",
                recovery="Ingest an ATLAS v6 source file first.",
            )
        return generation_id

    def require_ready_generation_id(self) -> str:
        """Return the active generation identifier or fail with guidance.

        Returns:
            (str) Active generation identifier.

        Raises:
            AtlasReviewError: No generation is active for retrieval.
        """
        generation_id = self.repository.active_generation_id()
        if generation_id is None:
            raise AtlasReviewError(
                code="no_active_generation",
                stage=STAGE_RETRIEVAL,
                message="this workspace has no active index bundle",
                recovery=("Run an explicit contextualized ingestion build first."),
            )
        return generation_id


def _timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()
