"""Explicit in-place migration from workspace schema version 2 to 3.

The version 3 schema widens the CHECK constraints on ``entities.kind`` and
``edges.relation`` so ATLAS 6.1.0 software, entity, and relationship records can
be stored. SQLite cannot alter a CHECK constraint, so both tables are rebuilt
and their rows are copied. Opening a workspace never runs this migration; the
operator must confirm it first.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from atlas_review.errors import STAGE_DATABASE, AtlasReviewError
from atlas_review.storage.schema import SCHEMA_VERSION

PREVIOUS_SCHEMA_VERSION = 2

_REBUILD_STATEMENTS = (
    """
    CREATE TABLE entities_v3 (
        generation_id TEXT NOT NULL REFERENCES generations(generation_id),
        entity_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN
            ('collection', 'matrix', 'tactic', 'technique', 'mitigation',
             'case_study', 'software', 'entity')),
        name TEXT NOT NULL,
        display_name TEXT NOT NULL,
        source_locator TEXT NOT NULL,
        payload_msgpack BLOB NOT NULL,
        PRIMARY KEY (generation_id, entity_id)
    )
    """,
    """
    INSERT INTO entities_v3 (generation_id, entity_id, kind, name, display_name,
        source_locator, payload_msgpack)
    SELECT generation_id, entity_id, kind, name, display_name, source_locator,
        payload_msgpack
    FROM entities
    """,
    "DROP TABLE entities",
    "ALTER TABLE entities_v3 RENAME TO entities",
    "CREATE INDEX entities_by_kind ON entities(generation_id, kind)",
    """
    CREATE TABLE edges_v3 (
        generation_id TEXT NOT NULL REFERENCES generations(generation_id),
        edge_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation TEXT NOT NULL CHECK (relation IN
            ('employs', 'mitigates', 'specializes', 'achieves', 'sequences',
             'uses', 'attributed-to', 'implements', 'reported-by')),
        step_id TEXT,
        source_locator TEXT NOT NULL,
        payload_msgpack BLOB NOT NULL,
        PRIMARY KEY (generation_id, edge_id),
        FOREIGN KEY (generation_id, source_id)
            REFERENCES entities(generation_id, entity_id),
        FOREIGN KEY (generation_id, target_id)
            REFERENCES entities(generation_id, entity_id)
    )
    """,
    """
    INSERT INTO edges_v3 (generation_id, edge_id, source_id, target_id, relation,
        step_id, source_locator, payload_msgpack)
    SELECT generation_id, edge_id, source_id, target_id, relation, step_id,
        source_locator, payload_msgpack
    FROM edges
    """,
    "DROP TABLE edges",
    "ALTER TABLE edges_v3 RENAME TO edges",
    "CREATE INDEX edges_from ON edges(generation_id, source_id, relation)",
    "CREATE INDEX edges_to ON edges(generation_id, target_id, relation)",
)


def stored_schema_version(database: Path) -> int | None:
    """Return the ``PRAGMA user_version`` of an existing workspace database.

    Args:
        database (Path): Workspace SQLite file.

    Returns:
        (int | None) Stored schema version, or ``None`` when the file is absent
        or not a readable database.
    """
    if not database.exists():
        return None
    try:
        connection = sqlite3.connect(database)
        try:
            row = connection.execute("PRAGMA user_version").fetchone()
        finally:
            connection.close()
    except sqlite3.DatabaseError:
        return None
    return int(row[0]) if row is not None else None


def migrate_workspace(database: Path) -> None:
    """Rebuild a version 2 workspace database in place at version 3.

    Args:
        database (Path): Workspace SQLite file.

    Raises:
        AtlasReviewError: The file is new, unsupported, or the rebuild fails a
            foreign-key check.
    """
    if not database.exists():
        raise _migration_error(
            "migration_missing_database", "the workspace database does not exist"
        )
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == SCHEMA_VERSION:
            return
        if version != PREVIOUS_SCHEMA_VERSION:
            raise _migration_error(
                "migration_unsupported_schema",
                f"workspace schema version {version} cannot be migrated in place",
            )
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.execute("BEGIN IMMEDIATE")
            for statement in _REBUILD_STATEMENTS:
                connection.execute(statement)
            _require_sound_relationships(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()


def _require_sound_relationships(connection: sqlite3.Connection) -> None:
    """Refuse a rebuilt database whose foreign keys do not resolve."""
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise _migration_error(
            "migration_failed",
            "the migrated workspace has broken relationships",
        )


def _migration_error(code: str, message: str) -> AtlasReviewError:
    """Build a database-stage migration error with a recovery hint."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_DATABASE,
        message=message,
        recovery=(
            "Create a new workspace, or migrate a version "
            f"{PREVIOUS_SCHEMA_VERSION} workspace explicitly."
        ),
    )
