"""Live provider tests against a real OpenAI-compatible endpoint.

These tests make real network requests and therefore stay out of the default
offline suite. Run them explicitly::

    FIREWORKS_API_KEY=... uv run pytest --llm -m llm

The default provider is Fireworks AI (`https://api.fireworks.ai/inference/v1`).
`ATLAS_REVIEW_LIVE_MODEL` selects another Fireworks model. The configured model
must support Chat Completions; the review tests use JSON-object mode and a
large output allowance, and the configuration sets `reasoning_effort` to
`medium`, because a reasoning model otherwise spends the whole allowance on
reasoning before emitting the report.

`ATLAS_REVIEW_MODEL_CACHE` optionally points at an existing FastEmbed model
directory, so a local run reuses downloaded artifacts instead of fetching them
into the test workspace again.

Both contextualization and review synthesis use the same configured
endpoint/model pair, matching the application's stored-provider contract.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.contracts import EvidencePacket, ReviewRequest
from atlas_review.corpus.atlas_v6 import ParsedCorpus, parse_source
from atlas_review.corpus.contextualize import (
    SYSTEM_PROMPT,
    build_user_payload,
    parse_chunk_context,
    validate_context,
)
from atlas_review.corpus.ingest import prepare_build
from atlas_review.inference.client import (
    ChatContextualizer,
    ProviderReply,
    build_contextualizer,
    build_reviewer,
)
from atlas_review.inference.review import AttemptPlan, ReviewOutcome, generate_review
from atlas_review.services.session import WorkspaceSession
from fakes import RecordingReviewStore
from packets import TECHNIQUE, default_packet, packet, passage

pytestmark = pytest.mark.llm

LIVE_BASE_URL = "https://api.fireworks.ai/inference/v1"
LIVE_API_KEY_ENV = "FIREWORKS_API_KEY"
DEFAULT_LIVE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
LIVE_OUTPUT_TOKENS = 32768
LIVE_CONTEXT_OUTPUT_TOKENS = 32768
LIVE_TIMEOUT_SECONDS = 180.0
ACQUIRED_AT = "2026-11-01T00:00:00+00:00"
ATLAS_ATTRIBUTION = "Apache License 2.0"
MODEL_CACHE_ENV = "ATLAS_REVIEW_MODEL_CACHE"
FIXTURE_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_atlas_v6.yaml"


@dataclass(slots=True)
class CountingReviewer:
    """Count real review requests through the configured adapter."""

    inner: ChatContextualizer
    replies: list[ProviderReply] = field(default_factory=list)

    def __call__(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        output_tokens: int,
        temperature: float | None,
    ) -> ProviderReply:
        """Forward one request and keep its reply."""
        reply = self.inner(
            system_prompt,
            user_payload,
            output_tokens=output_tokens,
            temperature=temperature,
        )
        self.replies.append(reply)
        return reply


@dataclass(frozen=True, slots=True)
class LiveStack:
    """One open live workspace with its counting review adapter."""

    session: WorkspaceSession
    reviewer: CountingReviewer


def _share_model_cache(root: Path) -> None:
    """Link an existing model cache into the test workspace when configured."""
    source = os.environ.get(MODEL_CACHE_ENV)
    if not source:
        return
    models = root / "models"
    if not models.exists():
        models.symlink_to(Path(source).resolve(), target_is_directory=True)


def _sparse_packet() -> EvidencePacket:
    """Return a technique-only packet with no supporting associations."""
    item = passage(
        "E001",
        entities=[TECHNIQUE],
        passage_id=1,
        text=(
            "Indirect prompt injection places instructions in data the model "
            "later retrieves."
        ),
    )
    return packet([item])


@pytest.fixture(scope="module")
def live_config() -> AppConfig:
    """Return the Fireworks provider configuration for the live run."""
    if not os.environ.get(LIVE_API_KEY_ENV, "").strip():
        pytest.skip(f"{LIVE_API_KEY_ENV} is not set for the live provider run")
    overrides = AppConfig().model_dump(mode="json")
    overrides["llm"] = {
        "provider_name": "fireworks",
        "base_url": LIVE_BASE_URL,
        "model": os.environ.get("ATLAS_REVIEW_LIVE_MODEL", DEFAULT_LIVE_MODEL),
        "api_key_env": LIVE_API_KEY_ENV,
        "timeout_seconds": LIVE_TIMEOUT_SECONDS,
        "output_tokens": LIVE_OUTPUT_TOKENS,
        "temperature": 0.0,
        "reasoning_effort": "medium",
        "response_format_mode": "json_object",
    }
    overrides["contextualization"] = {
        "output_tokens": LIVE_CONTEXT_OUTPUT_TOKENS,
        "temperature": 0.0,
    }
    return AppConfig.model_validate(overrides)


@pytest.fixture(scope="module")
def live_stack(
    live_config: AppConfig, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[LiveStack]:
    """Open one workspace storing the live provider configuration."""
    root = tmp_path_factory.mktemp("live-workspace")
    _share_model_cache(root)
    setup = WorkspaceSession(root)
    try:
        setup.save_config(live_config)
    finally:
        setup.close()
    reviewer = CountingReviewer(build_reviewer(live_config.llm))
    session = WorkspaceSession(root, reviewer=reviewer)
    try:
        yield LiveStack(session=session, reviewer=reviewer)
    finally:
        session.close()
        reviewer.inner.close()


@pytest.fixture(scope="module")
def live_corpus() -> ParsedCorpus:
    """Return the committed fixture parsed for a live build."""
    return parse_source(
        FIXTURE_SOURCE.read_bytes(),
        origin=str(FIXTURE_SOURCE),
        acquired_at=ACQUIRED_AT,
    )


@pytest.mark.parametrize(
    "owner_kind",
    [
        pytest.param("technique", id="technique"),
        pytest.param("mitigation", id="mitigation"),
        pytest.param("case_study", id="case-study"),
    ],
)
def test_live_contextualization_accepts_a_real_note(
    live_stack: LiveStack, live_corpus: ParsedCorpus, owner_kind: str
) -> None:
    """A real note passes locator, identifier, and token-budget validation."""
    session = live_stack.session
    counters = session.token_counters()
    prepared = prepare_build(live_corpus, config=session.config, counters=counters)
    chunk = next(
        candidate
        for candidate in prepared.chunks
        if candidate.owner_kind == owner_kind
        and candidate.section == "description"
        and candidate.ordinal == 0
    )
    document = prepared.documents[chunk.owner_id]
    contextualizer = build_contextualizer(session.config.llm)
    try:
        reply = contextualizer(
            SYSTEM_PROMPT,
            build_user_payload(document, chunk),
            output_tokens=session.config.contextualization.output_tokens,
            temperature=session.config.contextualization.temperature,
        )
    finally:
        contextualizer.close()

    accepted = validate_context(
        parse_chunk_context(reply.text),
        document=document,
        chunk=chunk,
        config=session.config,
        counters=counters,
    )
    assert set(accepted.support_locators) <= set(document.support_locators)
    assert (
        accepted.embedding_tokens <= session.config.contextualization.context_max_tokens
    )
    assert reply.model
    assert reply.response_id
    assert reply.input_tokens is not None
    assert reply.output_tokens is not None


@pytest.mark.parametrize(
    "evidence",
    [
        pytest.param(default_packet(), id="linked-packet"),
        pytest.param(_sparse_packet(), id="technique-only-packet"),
    ],
)
def test_live_review_synthesis_returns_a_validated_report(
    live_stack: LiveStack, review_brief: ReviewRequest, evidence: EvidencePacket
) -> None:
    """A real report passes strict citation and association validation."""
    store = RecordingReviewStore()
    outcome = generate_review(
        review_brief,
        evidence,
        live_stack.session.config.llm,
        plan=AttemptPlan(
            review_id="live-review",
            attempt_number=1,
            started_at=ACQUIRED_AT,
        ),
        reviewer=live_stack.reviewer,
        store=store,
    )
    attempt = store.attempts[-1]
    assert attempt.state == "complete"
    assert attempt.response_id
    assert attempt.usage_input_tokens is not None
    assert attempt.usage_output_tokens is not None
    assert len(outcome.report.findings) <= live_stack.session.config.llm.max_findings
    assert store.reports == [outcome.report]
    assert live_stack.reviewer.replies[-1].text == attempt.response_text


def test_live_end_to_end_review_is_saved_reopened_and_exported(
    live_stack: LiveStack, source_path: Path, review_brief: ReviewRequest
) -> None:
    """Ingest, retrieve, generate, reopen, and export without extra requests."""
    session = live_stack.session
    prepared = session.prepare_ingestion(source_path)
    assert prepared.plan.uncached_context_count == prepared.plan.chunk_count
    assert prepared.plan.endpoint_host == "api.fireworks.ai"

    built = session.build_generation(prepared)
    assert built.requested_context_count == prepared.plan.chunk_count
    assert built.passage_count == prepared.plan.chunk_count
    assert built.dense_count == built.keyword_count == built.passage_count

    reviews_before = len(live_stack.reviewer.replies)
    draft = session.retrieve_review(review_brief)
    assert len(live_stack.reviewer.replies) == reviews_before
    assert draft.run.selection.packet.items

    finished: ReviewOutcome = session.generate_report(draft)
    assert len(live_stack.reviewer.replies) == reviews_before + 1
    assert finished.attempt.state == "complete"

    stored = session.open_review(draft.review_id)
    assert stored.report == finished.report
    assert stored.evidence == draft.run.selection.packet

    exported = session.export_review(draft.review_id)
    markdown = exported.read_text(encoding="utf-8")
    assert stored.evidence.collection_version in markdown
    assert "## Evidence appendix" in markdown
    assert ATLAS_ATTRIBUTION in markdown
    generated = {
        stored_passage.context_text
        for stored_passage in session.passages.passages(stored.generation_id)
    }
    assert generated
    assert all(note not in markdown for note in generated)

    replayed = session.prepare_ingestion(source_path)
    assert replayed.plan.reuse_generation
    assert replayed.plan.uncached_context_count == 0
    reused = session.build_generation(replayed)
    assert reused.reused_generation
    assert reused.requested_context_count == 0
    assert session.repository.active_generation_id() == built.generation_id
