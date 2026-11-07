"""Shared fixtures for workspace, ingestion, and rendering tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from atlas_review.corpus.atlas_v6 import ParsedCorpus, parse_source
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.services.session import WorkspaceSession
from fakes import FixedContextualizer, RecordingEmbedder, WhitespaceTokenCounter

ACQUIRED_AT = "2026-11-01T00:00:00+00:00"

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


@pytest.fixture
def corpus(source_bytes: bytes, source_path: Path) -> ParsedCorpus:
    """Return the parsed committed fixture."""
    return parse_source(source_bytes, origin=str(source_path), acquired_at=ACQUIRED_AT)


@pytest.fixture
def counters() -> TokenCounters:
    """Return whitespace token counters standing in for the model tokenizers."""
    return TokenCounters(
        embedding=WhitespaceTokenCounter(), reranker=WhitespaceTokenCounter()
    )


@pytest.fixture
def embedder() -> RecordingEmbedder:
    """Return a recording hashed bag-of-words embedder."""
    return RecordingEmbedder()


@pytest.fixture
def contextualizer() -> FixedContextualizer:
    """Return a fixed chunk contextualizer."""
    return FixedContextualizer()
