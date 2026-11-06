"""Shared fixtures for workspace, ingestion, and rendering tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from atlas_review.services.session import WorkspaceSession

FIXTURE_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_atlas_v6.yaml"


@pytest.fixture
def source_path() -> Path:
    """Return the committed ATLAS v6 fixture path."""
    return FIXTURE_SOURCE


@pytest.fixture
def source_bytes(source_path: Path) -> bytes:
    """Return the exact bytes of the committed ATLAS v6 fixture."""
    return source_path.read_bytes()


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Return an empty workspace directory inside the test's temp directory."""
    return tmp_path / "workspace"


@pytest.fixture
def session(workspace_root: Path) -> Iterator[WorkspaceSession]:
    """Return an open workspace session that is closed after the test."""
    opened = WorkspaceSession(workspace_root)
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def stored_session(session: WorkspaceSession, source_path: Path) -> WorkspaceSession:
    """Return a session that already stores the committed fixture source."""
    session.ingest_file(source_path)
    return session
