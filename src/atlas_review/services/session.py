"""Workspace session: one configured workspace used by the terminal UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from atlas_review.config import AppConfig
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.ingest import (
    BuildServices,
    IngestionPlan,
    plan_ingestion,
    run_ingestion,
)
from atlas_review.corpus.ingest import IngestionOutcome as BuildOutcome
from atlas_review.errors import STAGE_INGESTION, STAGE_RETRIEVAL, AtlasReviewError
from atlas_review.inference.client import build_contextualizer
from atlas_review.retrieval.embeddings import EmbeddingBackend, FastEmbedBackend
from atlas_review.retrieval.hybrid import SearchOutcome, check_focus, search
from atlas_review.retrieval.loader import load_generation
from atlas_review.retrieval.tokenizers import TokenCounters, load_token_counters
from atlas_review.services.ingestion import (
    IngestionOutcome,
    SourcePreview,
    ingest_source,
    preview_source,
    read_source_file,
)
from atlas_review.services.settings import load_config
from atlas_review.storage.database import WorkspaceDatabase
from atlas_review.storage.passages import PassageRepository
from atlas_review.storage.repository import (
    CorpusRepository,
    EntityInspection,
    GenerationSummary,
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


class WorkspaceSession:
    """Coordinate one workspace database, configuration, and corpus repository."""

    def __init__(
        self,
        root: Path,
        *,
        counters: TokenCounters | None = None,
        embedding: EmbeddingBackend | None = None,
    ) -> None:
        """Open the workspace and load or initialise its configuration.

        Args:
            root (Path): Workspace directory supplied by ``--workspace``.
            counters (TokenCounters | None): Injected retrieval tokenizers.
            embedding (EmbeddingBackend | None): Injected embedding backend.
        """
        self.paths = WorkspacePaths.under(root)
        for directory in (self.paths.indexes, self.paths.models, self.paths.exports):
            directory.mkdir(parents=True, exist_ok=True)
        self.database = WorkspaceDatabase(self.paths.database)
        self.repository = CorpusRepository(self.database)
        self.passages = PassageRepository(self.database)
        self.config: AppConfig = load_config(self.database, updated_at=_timestamp())
        self._counters = counters
        self._embedding = embedding

    def close(self) -> None:
        """Close the workspace database connection."""
        self.database.close()

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
