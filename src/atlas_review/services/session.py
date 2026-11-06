"""Workspace session: one configured workspace used by the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from atlas_review.config import AppConfig
from atlas_review.errors import STAGE_INGESTION, AtlasReviewError
from atlas_review.services.ingestion import (
    IngestionOutcome,
    SourcePreview,
    ingest_source,
    preview_source,
    read_source_file,
)
from atlas_review.services.settings import load_config
from atlas_review.storage.database import WorkspaceDatabase
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


class WorkspaceSession:
    """Coordinate one workspace database, configuration, and corpus repository."""

    def __init__(self, root: Path) -> None:
        """Open the workspace and load or initialise its configuration.

        Args:
            root (Path): Workspace directory supplied by ``--workspace``.
        """
        self.paths = WorkspacePaths.under(root)
        for directory in (self.paths.indexes, self.paths.models, self.paths.exports):
            directory.mkdir(parents=True, exist_ok=True)
        self.database = WorkspaceDatabase(self.paths.database)
        self.repository = CorpusRepository(self.database)
        self.config: AppConfig = load_config(self.database, updated_at=_timestamp())

    def close(self) -> None:
        """Close the workspace database connection."""
        self.database.close()

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


def _timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()
