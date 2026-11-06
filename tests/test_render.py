"""Rendering behavior: sanitization and inspection output for untrusted text."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console, RenderableType

from atlas_review.services.ingestion import preview_source
from atlas_review.services.session import WorkspaceSession
from atlas_review.tui.render import (
    render_generations,
    render_inspection,
    render_preview,
    sanitize_terminal_text,
)


def _text(renderable: RenderableType) -> str:
    """Render one Rich renderable to plain text without colors."""
    console = Console(record=True, width=120, color_system=None)
    console.print(renderable)
    return console.export_text()


class TestSanitization:
    """Terminal controls never reach the terminal as escape sequences."""

    def test_escape_sequences_are_removed(self) -> None:
        assert sanitize_terminal_text("a\x1b[31mred\x1b[0mb") == "aredb"

    def test_control_characters_are_removed_but_newlines_survive(self) -> None:
        assert sanitize_terminal_text("one\ntwo\x07\x00") == "one\ntwo"


class TestPreviewRendering:
    """A source preview shows identity, counts, and sanitized warnings."""

    def test_preview_includes_digest_counts_and_warnings(
        self, source_bytes: bytes
    ) -> None:
        corpus, preview = preview_source(
            source_bytes, origin="/fixtures/mini_atlas_v6.yaml"
        )
        rendered = _text(render_preview(preview))
        assert "/fixtures/mini_atlas_v6.yaml" in rendered
        assert corpus.snapshot.source_sha256 in rendered
        assert "technique=3" in rendered
        assert "employs=3" in rendered
        assert "warnings" in rendered


class TestInspectionRendering:
    """Inspection output carries entity fields, steps, and provenance."""

    def test_case_inspection_shows_steps_and_locators(
        self, stored_session: WorkspaceSession
    ) -> None:
        rendered = _text(render_inspection(stored_session.inspect("AML.CS0035")))
        assert "AML.CS0035" in rendered
        assert "case_study" in rendered
        assert "S01" in rendered
        assert "AML.T0051.001" in rendered

    def test_mitigation_links_show_both_directions(
        self, stored_session: WorkspaceSession
    ) -> None:
        rendered = _text(render_inspection(stored_session.inspect("AML.M0033")))
        assert "Outgoing relationships" in rendered
        assert "mitigates" in rendered
        assert "AML.T0051" in rendered

    def test_generation_table_lists_stored_generation(
        self, stored_session: WorkspaceSession
    ) -> None:
        rendered = _text(render_generations(stored_session.generations()))
        assert stored_session.require_generation_id() in rendered
        assert "building" in rendered

    def test_empty_generation_list_explains_the_next_step(self) -> None:
        rendered = _text(render_generations(()))
        assert "No corpus generation" in rendered


def test_rendering_does_not_modify_stored_workspace(
    stored_session: WorkspaceSession, workspace_root: Path
) -> None:
    before = (workspace_root / "atlas.sqlite3").stat().st_size
    _text(render_inspection(stored_session.inspect("AML.T0051")))
    assert (workspace_root / "atlas.sqlite3").stat().st_size == before
