"""Corpus persistence and inspection queries over a workspace database."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal

from atlas_review.contracts import (
    AtlasEntity,
    AtlasRelationship,
    GenerationManifest,
    SourceSnapshot,
)
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.errors import STAGE_STORAGE, AtlasReviewError
from atlas_review.storage.codec import decode_model, encode_model
from atlas_review.storage.database import WorkspaceDatabase

GenerationStatus = Literal["building", "ready"]

_INSERT_GENERATION = """
INSERT INTO generations (
    generation_id, created_at, collection_version, format_version,
    source_sha256, build_fingerprint, source_bytes, manifest_msgpack, status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_ENTITY = """
INSERT INTO entities (
    generation_id, entity_id, kind, name, display_name, source_locator,
    payload_msgpack
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_EDGE = """
INSERT INTO edges (
    generation_id, edge_id, source_id, target_id, relation, step_id,
    source_locator, payload_msgpack
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Published bundle location recorded for a ready generation.

    Instance Attributes:
        directory (str): Bundle path relative to the workspace.
        manifest_sha256 (str): Digest of the bundle manifest.
    """

    directory: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Stored generation identity used to decide reuse or rebuild.

    Instance Attributes:
        generation_id (str): Stored generation identifier.
        status (GenerationStatus): ``building`` or ``ready``.
        build_fingerprint (str): Recorded completed build fingerprint.
    """

    generation_id: str
    status: GenerationStatus
    build_fingerprint: str


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Counts and identity for one stored corpus generation.

    Instance Attributes:
        generation_id (str): Deterministic generation identifier.
        created_at (str): ISO 8601 storage timestamp.
        collection_version (str): ATLAS collection version.
        format_version (str): Declared source format version.
        status (GenerationStatus): ``building`` or ``ready``.
        entity_count (int): Stored entity records.
        edge_count (int): Stored relationship records.
    """

    generation_id: str
    created_at: str
    collection_version: str
    format_version: str
    status: GenerationStatus
    entity_count: int
    edge_count: int


@dataclass(frozen=True, slots=True)
class EntityInspection:
    """One entity with its incoming and outgoing relationship provenance.

    Instance Attributes:
        entity (AtlasEntity): Canonical entity record.
        outgoing (tuple[AtlasRelationship, ...]): Edges owned by the entity.
        incoming (tuple[AtlasRelationship, ...]): Edges naming the entity.
    """

    entity: AtlasEntity
    outgoing: tuple[AtlasRelationship, ...]
    incoming: tuple[AtlasRelationship, ...]


class CorpusRepository:
    """Read and write corpus records for one workspace database."""

    def __init__(self, database: WorkspaceDatabase) -> None:
        """Bind the repository to a workspace database.

        Args:
            database (WorkspaceDatabase): Workspace connection owner.
        """
        self._database = database

    def store_generation(
        self,
        corpus: ParsedCorpus,
        manifest: GenerationManifest,
        source_bytes: bytes,
    ) -> None:
        """Insert one generation with its entities and edges atomically.

        The generation stays ``building`` until index publication; Increment 1
        stores no index bundle and therefore never activates a generation.

        Args:
            corpus (ParsedCorpus): Parsed records to persist.
            manifest (GenerationManifest): Recorded identity and counts.
            source_bytes (bytes): Exact source bytes for later provenance.

        Raises:
            AtlasReviewError: The generation identifier already exists.
        """
        snapshot = corpus.snapshot
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    _INSERT_GENERATION,
                    (
                        corpus.generation_id,
                        manifest.created_at,
                        snapshot.collection_version,
                        snapshot.format_version,
                        snapshot.source_sha256,
                        manifest.build_fingerprint,
                        source_bytes,
                        encode_model(manifest),
                        "building",
                    ),
                )
                _insert_entities(connection, corpus.generation_id, corpus.entities)
                _insert_edges(connection, corpus.generation_id, corpus.edges)
        except sqlite3.IntegrityError as exc:
            raise AtlasReviewError(
                code="generation_exists",
                stage=STAGE_STORAGE,
                message=f"generation {corpus.generation_id} is already stored",
                recovery="Inspect the stored generation or create a new workspace.",
            ) from exc

    def insert_generation(
        self,
        connection: sqlite3.Connection,
        corpus: ParsedCorpus,
        manifest: GenerationManifest,
        source_bytes: bytes,
    ) -> None:
        """Insert one generation row inside an existing transaction.

        Args:
            connection (sqlite3.Connection): Open immediate transaction.
            corpus (ParsedCorpus): Parsed records to persist.
            manifest (GenerationManifest): Recorded identity and counts.
            source_bytes (bytes): Exact source bytes for later provenance.
        """
        snapshot = corpus.snapshot
        connection.execute(
            _INSERT_GENERATION,
            (
                corpus.generation_id,
                manifest.created_at,
                snapshot.collection_version,
                snapshot.format_version,
                snapshot.source_sha256,
                manifest.build_fingerprint,
                source_bytes,
                encode_model(manifest),
                "building",
            ),
        )
        _insert_entities(connection, corpus.generation_id, corpus.entities)
        _insert_edges(connection, corpus.generation_id, corpus.edges)

    def finalize_generation(
        self,
        connection: sqlite3.Connection,
        *,
        manifest: GenerationManifest,
        artifact: StoredArtifact,
    ) -> None:
        """Mark a generation ready and activate its published bundle.

        Args:
            connection (sqlite3.Connection): Open immediate transaction.
            manifest (GenerationManifest): Completed manifest with artifact hash.
            artifact (StoredArtifact): Published bundle location and digest.
        """
        connection.execute(
            """
            UPDATE generations SET
                status = 'ready', artifact_directory = ?,
                artifact_manifest_sha256 = ?, passage_count = ?,
                manifest_msgpack = ?
            WHERE generation_id = ?
            """,
            (
                artifact.directory,
                artifact.manifest_sha256,
                manifest.passage_count,
                encode_model(manifest),
                manifest.generation_id,
            ),
        )
        connection.execute(
            "UPDATE workspace_state SET active_generation_id = ? WHERE singleton = 1",
            (manifest.generation_id,),
        )

    def activate(self, generation_id: str) -> None:
        """Point the workspace at an already ready generation atomically."""
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE workspace_state SET active_generation_id = ? "
                "WHERE singleton = 1",
                (generation_id,),
            )

    def update_artifact(
        self,
        connection: sqlite3.Connection,
        *,
        manifest: GenerationManifest,
        artifact: StoredArtifact,
    ) -> None:
        """Record a repaired bundle for an already ready generation.

        Args:
            connection (sqlite3.Connection): Open immediate transaction.
            manifest (GenerationManifest): Manifest recomputed for the bundle.
            artifact (StoredArtifact): New bundle location and manifest digest.

        Raises:
            AtlasReviewError: The generation is not ready or does not exist, so
                no row was updated.
        """
        cursor = connection.execute(
            """
            UPDATE generations SET
                artifact_directory = ?, artifact_manifest_sha256 = ?,
                passage_count = ?, manifest_msgpack = ?
            WHERE generation_id = ? AND status = 'ready'
            """,
            (
                artifact.directory,
                artifact.manifest_sha256,
                manifest.passage_count,
                encode_model(manifest),
                manifest.generation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise AtlasReviewError(
                code="generation_not_repairable",
                stage=STAGE_STORAGE,
                message=f"generation {manifest.generation_id} is not ready",
                recovery="Re-ingest the source or select a ready generation.",
            )

    def generation(self, generation_id: str) -> GenerationRecord | None:
        """Return the stored identity of one generation, if it exists."""
        row = (
            self._database.connect()
            .execute(
                "SELECT generation_id, status, build_fingerprint FROM generations "
                "WHERE generation_id = ?",
                (generation_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return GenerationRecord(
            generation_id=row["generation_id"],
            status=row["status"],
            build_fingerprint=row["build_fingerprint"],
        )

    def transaction(self) -> AbstractContextManager[sqlite3.Connection]:
        """Open one explicit immediate transaction on the workspace database."""
        return self._database.transaction()

    def artifact(self, generation_id: str) -> StoredArtifact | None:
        """Return the recorded bundle location of one generation."""
        row = self._generation_row(generation_id)
        directory = row["artifact_directory"]
        digest = row["artifact_manifest_sha256"]
        if directory is None or digest is None:
            return None
        return StoredArtifact(directory=str(directory), manifest_sha256=str(digest))

    def artifact_is_referenced(self, directory: str) -> bool:
        """Return whether any stored generation records this bundle path.

        Args:
            directory (str): Bundle path relative to the workspace.

        Returns:
            (bool) ``True`` when at least one generation row references it.
        """
        row = (
            self._database.connect()
            .execute(
                "SELECT 1 FROM generations WHERE artifact_directory = ? LIMIT 1",
                (directory,),
            )
            .fetchone()
        )
        return row is not None

    def active_generation_id(self) -> str | None:
        """Return the active generation identifier, if the workspace has one."""
        row = (
            self._database.connect()
            .execute(
                "SELECT active_generation_id FROM workspace_state WHERE singleton = 1"
            )
            .fetchone()
        )
        return None if row is None else row["active_generation_id"]

    def latest_generation_id(self) -> str | None:
        """Return the most recently created generation identifier."""
        row = (
            self._database.connect()
            .execute(
                "SELECT generation_id FROM generations ORDER BY rowid DESC LIMIT 1"
            )
            .fetchone()
        )
        return None if row is None else row["generation_id"]

    def summaries(self) -> tuple[GenerationSummary, ...]:
        """Return every stored generation, newest first."""
        rows = (
            self._database.connect()
            .execute(
                """
            SELECT g.generation_id, g.created_at, g.collection_version,
                   g.format_version, g.status,
                   (SELECT count(*) FROM entities e
                    WHERE e.generation_id = g.generation_id) AS entity_count,
                   (SELECT count(*) FROM edges d
                    WHERE d.generation_id = g.generation_id) AS edge_count
            FROM generations g
            ORDER BY g.rowid DESC
            """
            )
            .fetchall()
        )
        return tuple(
            GenerationSummary(
                generation_id=row["generation_id"],
                created_at=row["created_at"],
                collection_version=row["collection_version"],
                format_version=row["format_version"],
                status=row["status"],
                entity_count=int(row["entity_count"]),
                edge_count=int(row["edge_count"]),
            )
            for row in rows
        )

    def manifest(self, generation_id: str) -> GenerationManifest:
        """Decode and revalidate the stored generation manifest."""
        row = self._generation_row(generation_id)
        return decode_model(row["manifest_msgpack"], GenerationManifest)

    def snapshot(self, generation_id: str) -> SourceSnapshot:
        """Return the source identity recorded for a generation."""
        return self.manifest(generation_id).snapshot

    def entity_ids(
        self, generation_id: str, kind: str | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Return ``(entity_id, display_name)`` pairs, newest kinds first."""
        if kind is None:
            rows = (
                self._database.connect()
                .execute(
                    "SELECT entity_id, display_name FROM entities "
                    "WHERE generation_id = ? ORDER BY rowid",
                    (generation_id,),
                )
                .fetchall()
            )
        else:
            rows = (
                self._database.connect()
                .execute(
                    "SELECT entity_id, display_name FROM entities "
                    "WHERE generation_id = ? AND kind = ? ORDER BY rowid",
                    (generation_id, kind),
                )
                .fetchall()
            )
        return tuple((row["entity_id"], row["display_name"]) for row in rows)

    def entity(self, generation_id: str, entity_id: str) -> AtlasEntity:
        """Decode one entity or fail with a storage error."""
        row = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM entities "
                "WHERE generation_id = ? AND entity_id = ?",
                (generation_id, entity_id),
            )
            .fetchone()
        )
        if row is None:
            raise _unknown_entity(entity_id)
        return decode_model(row["payload_msgpack"], AtlasEntity)

    def entities(self, generation_id: str) -> tuple[AtlasEntity, ...]:
        """Return every stored entity of one generation in stable order.

        Args:
            generation_id (str): Stored generation identifier.

        Returns:
            (tuple[AtlasEntity, ...]) Validated entities ordered by identifier.
        """
        rows = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM entities WHERE generation_id = ? "
                "ORDER BY entity_id",
                (generation_id,),
            )
            .fetchall()
        )
        return tuple(decode_model(row["payload_msgpack"], AtlasEntity) for row in rows)

    def edges(self, generation_id: str) -> tuple[AtlasRelationship, ...]:
        """Return every stored relationship edge of one generation.

        Args:
            generation_id (str): Stored generation identifier.

        Returns:
            (tuple[AtlasRelationship, ...]) Validated edges ordered by ID.
        """
        rows = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM edges WHERE generation_id = ? "
                "ORDER BY edge_id",
                (generation_id,),
            )
            .fetchall()
        )
        return tuple(
            decode_model(row["payload_msgpack"], AtlasRelationship) for row in rows
        )

    def inspect(self, generation_id: str, entity_id: str) -> EntityInspection:
        """Return one entity together with both relationship directions."""
        entity = self.entity(generation_id, entity_id)
        return EntityInspection(
            entity=entity,
            outgoing=self.edges_from(generation_id, entity_id),
            incoming=self.edges_to(generation_id, entity_id),
        )

    def edges_from(
        self, generation_id: str, entity_id: str, relation: str | None = None
    ) -> tuple[AtlasRelationship, ...]:
        """Return outgoing edges of one entity, optionally of one relation."""
        return self._edges(
            "SELECT payload_msgpack FROM edges "
            "WHERE generation_id = ? AND source_id = ?",
            (generation_id, entity_id),
            relation,
        )

    def edges_to(
        self, generation_id: str, entity_id: str, relation: str | None = None
    ) -> tuple[AtlasRelationship, ...]:
        """Return incoming edges of one entity, optionally of one relation."""
        return self._edges(
            "SELECT payload_msgpack FROM edges "
            "WHERE generation_id = ? AND target_id = ?",
            (generation_id, entity_id),
            relation,
        )

    def _edges(
        self, statement: str, parameters: tuple[str, ...], relation: str | None
    ) -> tuple[AtlasRelationship, ...]:
        """Run one parameterized edge query and decode its payloads."""
        if relation is not None:
            statement = f"{statement} AND relation = ?"
            parameters = (*parameters, relation)
        rows = (
            self._database.connect()
            .execute(f"{statement} ORDER BY rowid", parameters)
            .fetchall()
        )
        return tuple(
            decode_model(row["payload_msgpack"], AtlasRelationship) for row in rows
        )

    def _generation_row(self, generation_id: str) -> sqlite3.Row:
        """Return the generation row or fail with a storage error."""
        row: sqlite3.Row | None = (
            self._database.connect()
            .execute(
                "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
            )
            .fetchone()
        )
        if row is None:
            raise AtlasReviewError(
                code="unknown_generation",
                stage=STAGE_STORAGE,
                message=f"generation {generation_id} is not stored",
                recovery="Ingest the source again or select a stored generation.",
            )
        return row


def _insert_entities(
    connection: sqlite3.Connection,
    generation_id: str,
    entities: Sequence[AtlasEntity],
) -> None:
    """Insert entity rows with validated MessagePack payloads."""
    connection.executemany(
        _INSERT_ENTITY,
        [
            (
                generation_id,
                entity.entity_id,
                entity.kind,
                entity.name,
                entity.display_name,
                entity.source_locator,
                encode_model(entity),
            )
            for entity in entities
        ],
    )


def _insert_edges(
    connection: sqlite3.Connection,
    generation_id: str,
    edges: Sequence[AtlasRelationship],
) -> None:
    """Insert edge rows with validated MessagePack payloads."""
    connection.executemany(
        _INSERT_EDGE,
        [
            (
                generation_id,
                edge.edge_id,
                edge.source_id,
                edge.target_id,
                edge.relation,
                edge.step_id,
                edge.source_locator,
                encode_model(edge),
            )
            for edge in edges
        ],
    )


def _unknown_entity(entity_id: str) -> AtlasReviewError:
    """Build the storage error for a missing entity identifier."""
    return AtlasReviewError(
        code="unknown_entity",
        stage=STAGE_STORAGE,
        message=f"entity {entity_id} is not stored in this generation",
        recovery="List stored entities and select an existing identifier.",
    )
