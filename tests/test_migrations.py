"""Explicit schema migration: a version 2 workspace becomes version 3 in place."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from rich.console import Console
from rich.prompt import Confirm

from atlas_review.cli import upgrade_workspace
from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.storage.migrations import (
    PREVIOUS_SCHEMA_VERSION,
    migrate_workspace,
    stored_schema_version,
)
from atlas_review.storage.schema import SCHEMA_VERSION

TABLES = {
    "generations": "SELECT count(*) FROM generations",
    "entities": "SELECT count(*) FROM entities",
    "edges": "SELECT count(*) FROM edges",
    "passages": "SELECT count(*) FROM passages",
    "reviews": "SELECT count(*) FROM reviews",
}


def _counts(database: Path) -> dict[str, int]:
    """Count rows in the tables a migration must preserve."""
    with sqlite3.connect(database) as connection:
        return {
            table: connection.execute(statement).fetchone()[0]
            for table, statement in TABLES.items()
        }


@pytest.fixture
def previous_workspace(workspace_root: Path, source_path: Path) -> Path:
    """Return a populated workspace marked as the previous schema version."""
    session = WorkspaceSession(workspace_root)
    try:
        session.ingest_file(source_path)
    finally:
        session.close()
    database = workspace_root / "atlas.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {PREVIOUS_SCHEMA_VERSION}")
    return workspace_root


def test_migration_preserves_every_stored_row(previous_workspace: Path) -> None:
    """The rebuild copies entities, edges, generations, and reviews."""
    database = previous_workspace / "atlas.sqlite3"
    before = _counts(database)
    migrate_workspace(database)
    assert stored_schema_version(database) == SCHEMA_VERSION
    assert _counts(database) == before
    assert before["entities"] > 0
    assert before["edges"] > 0
    session = WorkspaceSession(previous_workspace)
    try:
        assert len(session.generations()) == 1
        assert session.inspect("AML.T0051").entity.entity_id == "AML.T0051"
    finally:
        session.close()


@pytest.mark.usefixtures("loaded_generation")
def test_migration_preserves_stored_passages(
    fake_session: WorkspaceSession,
) -> None:
    """Rebuilding the parent tables keeps passage foreign keys intact."""
    database = fake_session.paths.database
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {PREVIOUS_SCHEMA_VERSION}")
    before = _counts(database)
    assert before["passages"] > 0
    migrate_workspace(database)
    assert stored_schema_version(database) == SCHEMA_VERSION
    assert _counts(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_is_idempotent(previous_workspace: Path) -> None:
    """A second call on a current workspace changes nothing."""
    database = previous_workspace / "atlas.sqlite3"
    migrate_workspace(database)
    before = _counts(database)
    migrate_workspace(database)
    assert stored_schema_version(database) == SCHEMA_VERSION
    assert _counts(database) == before


def test_opening_an_old_workspace_never_migrates_silently(
    previous_workspace: Path,
) -> None:
    """The session refuses an old schema until the operator confirms."""
    database = previous_workspace / "atlas.sqlite3"
    with pytest.raises(AtlasReviewError) as caught:
        WorkspaceSession(previous_workspace)
    assert caught.value.code == "unsupported_database_schema"
    assert stored_schema_version(database) == PREVIOUS_SCHEMA_VERSION


@pytest.mark.parametrize("version", [0, 1, SCHEMA_VERSION + 1])
def test_unsupported_schema_is_refused(tmp_path: Path, version: int) -> None:
    """Only the immediately previous schema can be migrated in place."""
    database = tmp_path / "atlas.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE foreign_table (id INTEGER)")
        connection.execute(f"PRAGMA user_version = {version}")
    with pytest.raises(AtlasReviewError) as caught:
        migrate_workspace(database)
    assert caught.value.code == "migration_unsupported_schema"


def test_missing_database_is_refused(tmp_path: Path) -> None:
    """A migration never creates a database."""
    with pytest.raises(AtlasReviewError) as caught:
        migrate_workspace(tmp_path / "absent.sqlite3")
    assert caught.value.code == "migration_missing_database"


def test_stored_schema_version_reads_or_reports_absent(tmp_path: Path) -> None:
    """The version probe is read-only and tolerates an absent file."""
    assert stored_schema_version(tmp_path / "absent.sqlite3") is None
    database = tmp_path / "atlas.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {PREVIOUS_SCHEMA_VERSION}")
    assert stored_schema_version(database) == PREVIOUS_SCHEMA_VERSION


def test_cli_declined_migration_leaves_the_workspace(
    previous_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining keeps the old schema and reports the choice."""
    monkeypatch.setattr(Confirm, "ask", lambda *_args, **_kwargs: False)
    database = previous_workspace / "atlas.sqlite3"
    before = _counts(database)
    with pytest.raises(AtlasReviewError) as caught:
        upgrade_workspace(Console(), previous_workspace)
    assert caught.value.code == "workspace_migration_declined"
    assert stored_schema_version(database) == PREVIOUS_SCHEMA_VERSION
    assert _counts(database) == before


def test_cli_confirmed_migration_upgrades_the_workspace(
    previous_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirming the prompt runs the in-place migration."""
    monkeypatch.setattr(Confirm, "ask", lambda *_args, **_kwargs: True)
    upgrade_workspace(Console(), previous_workspace)
    assert stored_schema_version(previous_workspace / "atlas.sqlite3") == SCHEMA_VERSION


def test_current_workspace_needs_no_prompt(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current workspace opens without asking anything."""
    session = WorkspaceSession(workspace_root)
    session.close()
    monkeypatch.setattr(
        Confirm,
        "ask",
        lambda *_args, **_kwargs: pytest.fail("unexpected prompt"),
    )
    upgrade_workspace(Console(), workspace_root)
