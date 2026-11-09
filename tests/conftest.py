"""Shared fixtures for workspace, ingestion, and rendering tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.contracts import ReviewRequest
from atlas_review.corpus.atlas_v6 import ParsedCorpus, parse_source
from atlas_review.corpus.ingest import BuildServices, run_ingestion
from atlas_review.retrieval.loader import LoadedGeneration, load_generation
from atlas_review.retrieval.pipeline import RetrievalRun, retrieve
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.services.session import WorkspaceSession
from atlas_review.storage.reviews import ReviewRetrieval, StoredReview
from fakes import (
    FixedContextualizer,
    RecordingEmbedder,
    RecordingReranker,
    RecordingReviewer,
    WhitespaceTokenCounter,
)
from packets import CREATED_AT, default_packet, sample_trace, valid_report

ACQUIRED_AT = "2026-11-01T00:00:00+00:00"
REVIEW_ID = "review-fixture"

FIXTURE_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_atlas_v6.yaml"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in live-provider switch."""
    parser.addoption(
        "--llm",
        action="store_true",
        default=False,
        help=(
            "run the live provider tests in tests/test_live_llm.py; the configured "
            "credential variable must be present in the environment"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker that opts one test into live provider calls."""
    config.addinivalue_line(
        "markers", "llm: makes live provider requests (opt-in with --llm)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Keep live provider tests out of the default offline suite."""
    live = config.getoption("llm")
    for item in items:
        if "llm" not in item.keywords:
            continue
        if live:
            item.add_marker(pytest.mark.enable_socket)
        else:
            item.add_marker(
                pytest.mark.skip(reason="live provider test; pass --llm to run it")
            )


RETRIEVAL_FOCUS = (
    "indirect prompt injection through retrieved workspace messages "
    "influencing sensitive answers"
)

BRIEF: dict[str, object] = {
    "schema_version": 1,
    "title": "Internal knowledge assistant design review",
    "review_question": (
        "How could untrusted retrieved content influence sensitive answers, "
        "and what should we verify before rollout?"
    ),
    "retrieval_focus": RETRIEVAL_FOCUS,
    "system_description": (
        "An employee assistant retrieves internal documents and workspace "
        "messages, then returns a cited answer under a service identity."
    ),
    "users_and_roles": ["Employees", "External collaborators"],
    "data_sources": ["Internal document repository", "Workspace messages"],
    "untrusted_inputs": ["Messages written by external collaborators"],
    "tools_and_permissions": ["Search documents", "Return an answer"],
    "sensitive_assets": ["Internal documents"],
    "existing_controls": ["Employees authenticate before use"],
    "constraints": ["No action tools beyond retrieval"],
    "unknowns": ["Whether returned links are validated outside the model"],
}


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
    root = tmp_path / "workspace"
    cache = os.environ.get("ATLAS_REVIEW_MODEL_CACHE")
    if cache:
        root.mkdir(parents=True, exist_ok=True)
        (root / "models").symlink_to(Path(cache).resolve(), target_is_directory=True)
    return root


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


@pytest.fixture
def reranker() -> RecordingReranker:
    """Return a recording, deterministic cross-encoder stand-in."""
    return RecordingReranker()


@pytest.fixture
def fake_session(
    workspace_root: Path, counters: TokenCounters, embedder: RecordingEmbedder
) -> Iterator[WorkspaceSession]:
    """Return an open session bound to the deterministic model fakes."""
    opened = WorkspaceSession(workspace_root, counters=counters, embedding=embedder)
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def loaded_generation(
    fake_session: WorkspaceSession,
    corpus: ParsedCorpus,
    counters: TokenCounters,
    embedder: RecordingEmbedder,
    contextualizer: FixedContextualizer,
) -> LoadedGeneration:
    """Build the fixture generation once and return its verified bundle."""
    run_ingestion(
        corpus,
        source_bytes=b"fixture",
        services=BuildServices(
            config=fake_session.config,
            repository=fake_session.repository,
            passages=fake_session.passages,
            indexes_dir=fake_session.paths.indexes,
            counters=counters,
            backend=embedder,
            contextualizer=contextualizer,
            progress=None,
        ),
    )
    return load_generation(
        fake_session.repository,
        passages=fake_session.passages,
        indexes_dir=fake_session.paths.indexes,
        config=fake_session.config,
        generation_id=fake_session.require_ready_generation_id(),
    )


@pytest.fixture
def retrieval_run(
    loaded_generation: LoadedGeneration,
    embedder: RecordingEmbedder,
    reranker: RecordingReranker,
    counters: TokenCounters,
) -> RetrievalRun:
    """Return one local retrieval run over the built fixture generation."""
    return retrieve(
        loaded_generation,
        embedder,
        reranker,
        AppConfig(),
        RETRIEVAL_FOCUS,
        counters,
    )


@pytest.fixture
def reviewer() -> RecordingReviewer:
    """Return a recording reviewer that reports insufficient evidence."""
    return RecordingReviewer()


@pytest.fixture
def review_session(
    workspace_root: Path,
    corpus: ParsedCorpus,
    counters: TokenCounters,
    embedder: RecordingEmbedder,
    contextualizer: FixedContextualizer,
    reviewer: RecordingReviewer,
) -> Iterator[WorkspaceSession]:
    """Return an open session with a ready fake-built generation and a reviewer."""
    opened = WorkspaceSession(
        workspace_root,
        counters=counters,
        embedding=embedder,
        reranker=RecordingReranker(),
        reviewer=reviewer,
    )
    try:
        run_ingestion(
            corpus,
            source_bytes=b"fixture",
            services=BuildServices(
                config=opened.config,
                repository=opened.repository,
                passages=opened.passages,
                indexes_dir=opened.paths.indexes,
                counters=counters,
                backend=embedder,
                contextualizer=contextualizer,
                progress=None,
            ),
        )
        yield opened
    finally:
        opened.close()


@pytest.fixture
def review_brief() -> ReviewRequest:
    """Return the documented internal-assistant brief."""
    return ReviewRequest.model_validate(BRIEF, strict=True)


@pytest.fixture
def stored_review(
    stored_session: WorkspaceSession, review_brief: ReviewRequest
) -> StoredReview:
    """Return one complete saved review over the synthetic packet."""
    generation_id = stored_session.require_generation_id()
    stored_session.reviews.save_retrieved(
        ReviewRetrieval(
            review_id=REVIEW_ID,
            generation_id=generation_id,
            created_at=CREATED_AT,
            request=review_brief,
            config=stored_session.config,
            trace=sample_trace(generation_id),
            evidence=default_packet(),
        )
    )
    stored_session.reviews.complete(REVIEW_ID, valid_report())
    return stored_session.reviews.review(REVIEW_ID)
