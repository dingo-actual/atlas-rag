"""End-to-end review flow through the workspace session.

These are adapter-level checks: the real SQLite review tables, the real prompt
builder, the real validator, and a recording stand-in for the provider.
"""

from __future__ import annotations

import pytest

from atlas_review.config import AppConfig, apply_provider_settings
from atlas_review.contracts import ReviewAttempt, ReviewRequest
from atlas_review.errors import AtlasReviewError
from atlas_review.inference.client import ProviderReply
from atlas_review.services.session import ReviewDraft, WorkspaceSession
from fakes import RecordingReviewer
from packets import CREATED_AT


class TestRetrievedDraft:
    """Retrieval freezes the brief and evidence without a provider call."""

    def test_draft_is_stored_before_generation(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        stored = review_session.open_review(draft.review_id)
        assert stored.state == "retrieved"
        assert stored.report is None
        assert stored.evidence == draft.run.selection.packet
        assert stored.request == review_brief
        assert [summary.review_id for summary in review_session.saved_reviews()] == [
            draft.review_id
        ]

    def test_draft_exposes_the_evidence_packet(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        assert draft.run.selection.packet.items
        assert draft.run.trace.evidence_ids


class TestGeneration:
    """One Generate action makes one request and stores its outcome."""

    def test_successful_generation_completes_the_review(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
        reviewer: RecordingReviewer,
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        outcome = review_session.generate_report(draft)
        stored = review_session.open_review(draft.review_id)
        assert stored.state == "complete"
        assert stored.report == outcome.report
        assert len(reviewer.calls) == 1
        assert [
            attempt.state for attempt in review_session.review_attempts(draft.review_id)
        ] == ["complete"]

    def test_reopening_a_report_makes_no_request(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
        reviewer: RecordingReviewer,
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        review_session.generate_report(draft)
        calls = len(reviewer.calls)
        reopened = review_session.open_review(draft.review_id)
        assert reopened.report is not None
        assert len(reviewer.calls) == calls

    def test_failed_attempt_keeps_the_evidence_and_allows_a_retry(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
        reviewer: RecordingReviewer,
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        reviewer.text = "not json"
        with pytest.raises(AtlasReviewError) as caught:
            review_session.generate_report(draft)
        assert caught.value.code == "invalid_report_json"
        failed = review_session.open_review(draft.review_id)
        assert failed.state == "failed"
        assert failed.evidence == draft.run.selection.packet
        reviewer.text = '{"schema_version": 1, "summary": "insufficient evidence"}'
        review_session.generate_report(draft)
        stored = review_session.open_review(draft.review_id)
        assert stored.state == "complete"
        assert [
            attempt.attempt_number
            for attempt in review_session.review_attempts(draft.review_id)
        ] == [1, 2]
        assert [
            attempt.state for attempt in review_session.review_attempts(draft.review_id)
        ] == ["failed", "complete"]

    def test_retry_after_success_keeps_the_accepted_report(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
        reviewer: RecordingReviewer,
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        review_session.generate_report(draft)
        accepted = review_session.open_review(draft.review_id).report
        reviewer.text = "still not json"
        with pytest.raises(AtlasReviewError):
            review_session.generate_report(draft)
        stored = review_session.open_review(draft.review_id)
        assert stored.state == "complete"
        assert stored.report == accepted

    def test_generation_without_credentials_preserves_the_draft(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ATLAS_REVIEW_API_KEY", raising=False)
        draft = review_session.retrieve_review(review_brief)
        offline = WorkspaceSession(review_session.paths.root)
        try:
            with pytest.raises(AtlasReviewError) as caught:
                offline.generate_report(draft)
            assert caught.value.stage == "llm_authentication"
            assert offline.open_review(draft.review_id).evidence
        finally:
            offline.close()


class TestExportAndConfiguration:
    """A saved review exports from storage, and provider settings persist."""

    def test_export_writes_into_the_workspace(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        review_session.generate_report(draft)
        target = review_session.export_review(draft.review_id)
        assert target.parent == review_session.paths.exports
        assert "Draft review based on the supplied application brief" in (
            target.read_text(encoding="utf-8")
        )

    def test_provider_settings_survive_a_reopen(
        self, review_session: WorkspaceSession
    ) -> None:
        provider = (
            '{"provider_name": "local", "base_url": "https://provider.example/v1",'
            ' "model": "review-model", "api_key_mode": "none", "api_key_env": null}'
        )
        review_session.save_config(
            apply_provider_settings(review_session.config, provider)
        )
        reopened = WorkspaceSession(review_session.paths.root)
        try:
            assert reopened.config.llm.model == "review-model"
            assert reopened.config.llm.api_key_mode == "none"
            assert reopened.config.retrieval == review_session.config.retrieval
        finally:
            reopened.close()


class TestRequestBudget:
    """The prepared request is what the screen reports before sending."""

    def test_prepared_request_uses_the_stored_provider_settings(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        review_session.save_config(
            AppConfig.model_validate(
                {
                    **review_session.config.model_dump(mode="json"),
                    "llm": {
                        **review_session.config.llm.model_dump(mode="json"),
                        "max_findings": 2,
                    },
                },
                strict=True,
            )
        )
        draft = review_session.retrieve_review(review_brief)
        request = review_session.prepare_review_request(draft)
        assert "at most 2 findings" in request.system
        assert request.packet == draft.run.selection.packet

    def test_prepared_request_carries_the_frozen_evidence(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        request = review_session.prepare_review_request(draft)
        assert "Context (generated)" not in request.user
        assert all(item.evidence_id in request.user for item in request.packet.items)


class TestGeneratedContextInspection:
    """A stored passage exposes its cached generated note for inspection."""

    def test_passage_context_and_cache_entry_are_returned(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        item = next(
            item
            for item in draft.run.selection.packet.items
            if item.passage_id is not None
        )
        assert item.passage_id is not None
        stored = review_session.generated_context(
            draft.run.selection.packet.generation_id, item.passage_id
        )
        assert stored.passage.passage_id == item.passage_id
        assert stored.passage.source_text in item.text
        assert stored.entry is not None
        assert stored.entry.context_text == stored.passage.context_text


class TestAttemptRecovery:
    """Abandoned attempts are reconciled and never submitted again."""

    def test_reopening_the_workspace_marks_a_started_attempt_interrupted(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        review_session.reviews.record(
            _attempt(draft.review_id, attempt_number=1, state="started")
        )
        review_session.close()
        reopened = WorkspaceSession(review_session.paths.root)
        try:
            attempts = reopened.review_attempts(draft.review_id)
            assert [attempt.state for attempt in attempts] == ["interrupted"]
            assert reopened.open_review(draft.review_id).state == "retrieved"
            assert reopened.reviews.next_attempt_number(draft.review_id) == 2
        finally:
            reopened.close()

    def test_interrupted_generation_keeps_the_evidence_for_a_retry(
        self, review_session: WorkspaceSession, review_brief: ReviewRequest
    ) -> None:
        draft = review_session.retrieve_review(review_brief)
        interrupting = WorkspaceSession(
            review_session.paths.root, reviewer=InterruptingReviewer()
        )
        try:
            with pytest.raises(KeyboardInterrupt):
                interrupting.generate_report(draft)
            assert [
                attempt.state
                for attempt in interrupting.review_attempts(draft.review_id)
            ] == ["interrupted"]
            stored = interrupting.open_review(draft.review_id)
            assert stored.state == "retrieved"
            assert stored.evidence == draft.run.selection.packet
            assert stored.report is None
        finally:
            interrupting.close()


def _attempt(review_id: str, *, attempt_number: int, state: str) -> ReviewAttempt:
    """Build one stored attempt record in the requested state."""
    return ReviewAttempt.model_validate(
        {
            "schema_version": 1,
            "review_id": review_id,
            "attempt_number": attempt_number,
            "state": state,
            "started_at": CREATED_AT,
            "provider_name": "custom-compatible",
            "base_url": "https://provider.example/v1",
            "model": "fake-model-1",
            "output_token_parameter": "max_tokens",
            "output_tokens": 2500,
            "response_format_mode": "text",
            "max_findings": 5,
            "request_characters": 9000,
            "evidence_item_count": 7,
        },
        strict=True,
    )


class InterruptingReviewer:
    """Abandon the provider call the way a terminal interrupt would."""

    def __call__(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        output_tokens: int,
        temperature: float | None,
    ) -> ProviderReply:
        """Raise the interrupt instead of answering."""
        del system_prompt, user_payload, output_tokens, temperature
        raise KeyboardInterrupt


def test_review_draft_dataclass_keeps_its_packet(
    review_session: WorkspaceSession, review_brief: ReviewRequest
) -> None:
    draft: ReviewDraft = review_session.retrieve_review(review_brief)
    assert isinstance(draft.brief, ReviewRequest)
    assert draft.run.selection.packet.generation_id == (
        review_session.require_ready_generation_id()
    )


def test_app_config_snapshot_is_stored_without_secrets(
    review_session: WorkspaceSession, review_brief: ReviewRequest
) -> None:
    draft = review_session.retrieve_review(review_brief)
    stored = review_session.open_review(draft.review_id)
    snapshot = AppConfig.model_validate(
        stored.config.model_dump(mode="json"), strict=True
    )
    assert snapshot == stored.config
    assert stored.config.llm.api_key_env == review_session.config.llm.api_key_env
