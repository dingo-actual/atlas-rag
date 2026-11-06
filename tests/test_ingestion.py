"""Ingestion service and CLI behavior: source limits, failures, and exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_review.cli import build_parser, main
from atlas_review.errors import AtlasReviewError
from atlas_review.services.ingestion import read_source_file
from atlas_review.services.session import WorkspaceSession


class TestSourceReading:
    """Source reading enforces the configured size and content limits."""

    def test_missing_file_reports_source_acquisition_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            read_source_file(tmp_path / "absent.yaml", max_bytes=1024)
        assert caught.value.code == "missing_source"
        assert caught.value.stage == "source_acquisition"

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_bytes(b"")
        with pytest.raises(AtlasReviewError) as caught:
            read_source_file(path, max_bytes=1024)
        assert caught.value.code == "empty_source"

    def test_file_above_limit_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "big.yaml"
        path.write_bytes(b"x" * 33)
        with pytest.raises(AtlasReviewError) as caught:
            read_source_file(path, max_bytes=32)
        assert caught.value.code == "source_too_large"

    def test_file_at_limit_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "edge.yaml"
        path.write_bytes(b"x" * 32)
        assert read_source_file(path, max_bytes=32) == b"x" * 32

    def test_preview_does_not_store_a_generation(
        self, session: WorkspaceSession, source_path: Path
    ) -> None:
        preview = session.preview_file(source_path)
        assert preview.byte_count > 0
        assert session.generations() == ()


class TestSourceParsing:
    """Invalid documents fail with the source-schema stage and no stored rows."""

    def test_unsupported_format_major_is_refused_without_storage(
        self, session: WorkspaceSession
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            session.ingest_file(_write(session.paths.root, b"format-version: 5.0.0\n"))
        assert caught.value.stage == "source_schema"
        assert session.generations() == ()

    def test_unreadable_yaml_is_refused_without_storage(
        self, session: WorkspaceSession
    ) -> None:
        with pytest.raises(AtlasReviewError):
            session.ingest_file(_write(session.paths.root, b"{{{"))
        assert session.generations() == ()


class TestCommandLine:
    """The entry point ingests one source and reports status without the menu."""

    def test_source_option_stores_generation_and_exits_zero(
        self, workspace_root: Path, source_path: Path
    ) -> None:
        status = main(
            ["--workspace", str(workspace_root), "--source", str(source_path)]
        )
        assert status == 0
        reopened = WorkspaceSession(workspace_root)
        try:
            assert len(reopened.generations()) == 1
            assert reopened.current_generation_id() is not None
        finally:
            reopened.close()

    def test_missing_source_exits_one(self, workspace_root: Path) -> None:
        status = main(
            ["--workspace", str(workspace_root), "--source", "absent-source.yaml"]
        )
        assert status == 1

    def test_defaults_target_relative_workspace_and_no_source(self) -> None:
        arguments = build_parser().parse_args([])
        assert arguments.workspace == Path("./.atlas-review")
        assert arguments.source is None


def _write(root: Path, content: bytes) -> Path:
    """Write an invalid source file inside the workspace root for one test."""
    path = root / "invalid-source.yaml"
    path.write_bytes(content)
    return path
