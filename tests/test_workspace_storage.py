"""Workspace storage behavior: schema, ingestion, inspection, and failures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.services.settings import load_config, save_config
from atlas_review.storage.database import WorkspaceDatabase
from atlas_review.storage.schema import SCHEMA_VERSION


class TestFreshWorkspace:
    """A new workspace initializes the documented layout and defaults."""

    def test_initialization_creates_layout_and_empty_generation_list(
        self, workspace_root: Path, session: WorkspaceSession
    ) -> None:
        assert workspace_root.joinpath("atlas.sqlite3").exists()
        assert session.paths.indexes.is_dir()
        assert session.paths.models.is_dir()
        assert session.paths.exports.is_dir()
        assert session.database.schema_version() == SCHEMA_VERSION
        assert session.config == AppConfig()
        assert session.generations() == ()
        assert session.current_generation_id() is None

    def test_inspection_without_generation_reports_ingestion_guidance(
        self, session: WorkspaceSession
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            session.inspect("AML.T0051")
        assert caught.value.code == "no_generation"
        assert caught.value.recovery


class TestIngestedFixture:
    """The committed fixture stores records that inspection can read back."""

    def test_store_records_summary_counts(
        self, stored_session: WorkspaceSession
    ) -> None:
        summaries = stored_session.generations()
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.status == "building"
        assert summary.collection_version == "2026.09"
        assert summary.format_version == "6.0.0"
        assert summary.entity_count == 10
        assert summary.edge_count == 8
        assert stored_session.current_generation_id() == summary.generation_id

    def test_technique_inspection_lists_both_edge_directions(
        self, stored_session: WorkspaceSession
    ) -> None:
        inspection = stored_session.inspect("AML.T0051")
        assert inspection.entity.kind == "technique"
        assert inspection.entity.display_name == "LLM Prompt Injection"
        assert inspection.outgoing == ()
        assert [(edge.relation, edge.source_id) for edge in inspection.incoming] == [
            ("mitigates", "AML.M0033"),
            ("specializes", "AML.T0051.001"),
        ]

    def test_case_step_inspection_preserves_step_provenance(
        self, stored_session: WorkspaceSession
    ) -> None:
        inspection = stored_session.inspect("AML.CS0035")
        assert inspection.entity.kind == "case_study"
        steps = [edge for edge in inspection.outgoing if edge.relation == "employs"]
        assert [edge.step_id for edge in steps] == ["S01", "S02", "S03"]
        assert [edge.position for edge in steps] == [0, 1, 2]
        assert steps[0].leads_to == ["S02"]
        assert all(edge.source_locator.startswith("/") for edge in steps)

    def test_entity_choices_are_limited_to_one_kind(
        self, stored_session: WorkspaceSession
    ) -> None:
        techniques = stored_session.entity_choices("technique")
        assert [entity_id for entity_id, _ in techniques] == [
            "AML.T0051",
            "AML.T0051.001",
            "AML.T0012",
        ]

    def test_unknown_entity_reports_storage_error(
        self, stored_session: WorkspaceSession
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            stored_session.inspect("AML.T9999")
        assert caught.value.code == "unknown_entity"
        assert caught.value.stage == "storage"

    def test_reingesting_identical_source_keeps_one_generation(
        self, stored_session: WorkspaceSession, source_path: Path, source_bytes: bytes
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            stored_session.ingest_file(source_path)
        assert caught.value.code == "generation_exists"
        assert len(stored_session.generations()) == 1
        assert (
            stored_session.database.connect()
            .execute("SELECT count(*) FROM generations")
            .fetchone()[0]
            == 1
        )
        assert source_bytes


class TestSchemaSafety:
    """Unsupported databases are refused; failed transactions leave no rows."""

    def test_newer_schema_version_is_refused_without_modification(
        self, workspace_root: Path
    ) -> None:
        workspace_root.mkdir(parents=True, exist_ok=True)
        path = workspace_root / "atlas.sqlite3"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE foreign_table (id INTEGER)")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(AtlasReviewError) as caught:
            WorkspaceDatabase(path).connect()
        assert caught.value.code == "unsupported_database_schema"
        with sqlite3.connect(path) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        assert [name for (name,) in tables] == ["foreign_table"]

    def test_failed_transaction_rolls_back_every_row(
        self, workspace_root: Path
    ) -> None:
        database = WorkspaceDatabase(workspace_root / "atlas.sqlite3")
        database.connect()

        def insert_then_fail(connection: sqlite3.Connection) -> None:
            connection.execute(
                "INSERT INTO configuration "
                "(singleton, schema_version, updated_at, payload_msgpack) "
                "VALUES (1, 2, 'now', x'00')"
            )
            raise RuntimeError("write failed")

        with pytest.raises(RuntimeError), database.transaction() as connection:
            insert_then_fail(connection)
        rows = (
            database.connect()
            .execute("SELECT count(*) FROM configuration")
            .fetchone()[0]
        )
        assert rows == 0
        database.close()


class TestStoredConfiguration:
    """The configuration table round-trips validated models only."""

    def test_replaced_configuration_is_returned_on_reload(
        self, workspace_root: Path
    ) -> None:
        database = WorkspaceDatabase(workspace_root / "atlas.sqlite3")
        default = load_config(database, updated_at="2026-11-06T00:00:00+00:00")
        changed = AppConfig.model_validate(
            {
                **default.model_dump(mode="python"),
                "corpus": {
                    **default.corpus.model_dump(mode="python"),
                    "max_source_bytes": 4_000_000,
                },
            }
        )
        save_config(database, changed, updated_at="2026-11-07T00:00:00+00:00")
        assert load_config(database, updated_at="2026-11-08T00:00:00+00:00") == changed
        database.close()

    def test_unsupported_stored_schema_version_is_refused(
        self, workspace_root: Path
    ) -> None:
        database = WorkspaceDatabase(workspace_root / "atlas.sqlite3")
        load_config(database, updated_at="2026-11-06T00:00:00+00:00")
        with database.transaction() as connection:
            connection.execute("UPDATE configuration SET schema_version = 99")
        with pytest.raises(AtlasReviewError) as caught:
            load_config(database, updated_at="2026-11-07T00:00:00+00:00")
        assert caught.value.code == "unsupported_configuration_schema"
        database.close()
