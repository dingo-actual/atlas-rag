"""Review synthesis: one explicit attempt, strict validation, recorded outcome."""

from __future__ import annotations

import json

import pytest

from atlas_review.config import LLMConfig
from atlas_review.contracts import ReviewRequest
from atlas_review.errors import (
    STAGE_LLM_PROVIDER,
    AtlasReviewError,
)
from atlas_review.inference.client import (
    ChatContextualizer,
    MissingProviderCredentialError,
    build_reviewer,
)
from atlas_review.inference.review import AttemptPlan, ReviewOutcome, generate_review
from atlas_review.inference.validate import MAX_REPORT_BYTES
from fakes import RecordingReviewer, RecordingReviewStore
from packets import default_packet, finding, valid_report

STARTED_AT = "2026-11-07T12:00:00+00:00"
REVIEW_ID = "review-1"


def provider(**overrides: object) -> LLMConfig:
    """Return a provider configuration with explicit overrides."""
    return LLMConfig.model_validate(
        {
            **LLMConfig(
                base_url="https://provider.example/v1", model="fake-model-1"
            ).model_dump(mode="json"),
            **overrides,
        },
        strict=True,
    )


def synthesize(
    brief: ReviewRequest,
    *,
    reviewer: RecordingReviewer,
    store: RecordingReviewStore | None = None,
    config: LLMConfig | None = None,
    attempt_number: int = 1,
) -> tuple[ReviewOutcome, RecordingReviewStore]:
    """Run one synthesis attempt against the default packet."""
    recorded = store or RecordingReviewStore()
    outcome = generate_review(
        brief,
        default_packet(),
        config or provider(),
        plan=AttemptPlan(
            review_id=REVIEW_ID,
            attempt_number=attempt_number,
            started_at=STARTED_AT,
        ),
        reviewer=reviewer,
        store=recorded,
    )
    return outcome, recorded


class TestSuccessfulAttempt:
    """A grounded report is stored with its finished attempt."""

    def test_report_and_attempt_are_recorded(self, review_brief: ReviewRequest) -> None:
        reviewer = RecordingReviewer(text=valid_report().model_dump_json())
        outcome, store = synthesize(review_brief, reviewer=reviewer)
        assert outcome.report == valid_report()
        assert store.reports == [valid_report()]
        assert [attempt.state for attempt in store.attempts] == ["started", "complete"]
        assert store.failures == 0

    def test_finished_attempt_records_usage_and_identity(
        self, review_brief: ReviewRequest
    ) -> None:
        reviewer = RecordingReviewer(text=valid_report().model_dump_json())
        _, store = synthesize(review_brief, reviewer=reviewer)
        finished = store.attempts[-1]
        assert finished.review_id == REVIEW_ID
        assert finished.attempt_number == 1
        assert finished.response_id == "resp-1"
        assert finished.usage_input_tokens == 1000
        assert finished.usage_output_tokens == 200
        assert finished.duration_seconds == 1.25
        assert finished.response_text == valid_report().model_dump_json()
        assert finished.error_code is None

    def test_request_uses_the_configured_output_options(
        self, review_brief: ReviewRequest
    ) -> None:
        reviewer = RecordingReviewer(text=valid_report().model_dump_json())
        config = provider(output_tokens=512, temperature=0.2, max_findings=3)
        synthesize(review_brief, reviewer=reviewer, config=config)
        system, user, output_tokens, temperature = reviewer.calls[0]
        assert output_tokens == 512
        assert temperature == 0.2
        assert "at most 3 findings" in system
        assert json.loads(user)["brief"]["title"] == review_brief.title

    def test_attempt_number_is_kept_for_a_retry(
        self, review_brief: ReviewRequest
    ) -> None:
        reviewer = RecordingReviewer(text=valid_report().model_dump_json())
        _, store = synthesize(review_brief, reviewer=reviewer, attempt_number=3)
        assert {attempt.attempt_number for attempt in store.attempts} == {3}


class TestFailedAttempt:
    """A failure saves a structured attempt and preserves the evidence."""

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            pytest.param("not json", "invalid_report_json", id="invalid-json"),
            pytest.param(
                "x" * (MAX_REPORT_BYTES + 1), "oversized_report", id="oversized"
            ),
            pytest.param(
                valid_report(finding(evidence_ids=["E999"])).model_dump_json(),
                "unsupported_report_citation",
                id="unknown-citation",
            ),
            pytest.param(
                valid_report(finding(technique_ids=["AML.T9999"])).model_dump_json(),
                "unsupported_report_citation",
                id="unknown-technique",
            ),
        ],
    )
    def test_invalid_output_is_recorded_without_a_report(
        self, review_brief: ReviewRequest, text: str, code: str
    ) -> None:
        reviewer = RecordingReviewer(text=text)
        store = RecordingReviewStore()
        with pytest.raises(AtlasReviewError) as caught:
            synthesize(review_brief, reviewer=reviewer, store=store)
        assert caught.value.code == code
        assert store.reports == []
        assert store.failures == 1
        assert [attempt.state for attempt in store.attempts] == ["started", "failed"]
        failed = store.attempts[-1]
        assert failed.error_code == code
        assert failed.response_text == text

    def test_provider_failure_is_recorded_and_reraised(
        self, review_brief: ReviewRequest
    ) -> None:
        error = AtlasReviewError(
            code="provider_request_failed",
            stage=STAGE_LLM_PROVIDER,
            message="the provider request failed: APITimeoutError",
        )
        reviewer = RecordingReviewer(error=error)
        store = RecordingReviewStore()
        with pytest.raises(AtlasReviewError) as caught:
            synthesize(review_brief, reviewer=reviewer, store=store)
        assert caught.value.code == "provider_request_failed"
        assert store.failures == 1
        assert store.attempts[-1].error_code == "provider_request_failed"
        assert store.attempts[-1].response_text is None

    def test_failure_details_name_the_field_without_the_payload(
        self, review_brief: ReviewRequest
    ) -> None:
        reviewer = RecordingReviewer(text='{"schema_version": 1}')
        with pytest.raises(AtlasReviewError) as caught:
            synthesize(review_brief, reviewer=reviewer)
        details = " ".join(caught.value.details)
        assert "summary" in details
        assert "{" not in details

    @pytest.mark.parametrize("limit", [1, 64, 500])
    def test_oversized_request_is_rejected_before_any_call(
        self, review_brief: ReviewRequest, limit: int
    ) -> None:
        reviewer = RecordingReviewer(text=valid_report().model_dump_json())
        store = RecordingReviewStore()
        with pytest.raises(AtlasReviewError) as caught:
            synthesize(
                review_brief,
                reviewer=reviewer,
                store=store,
                config=provider(max_input_characters=limit),
            )
        assert caught.value.code == "review_request_too_large"
        assert reviewer.calls == []
        assert store.attempts == []


class TestReviewerAdapter:
    """Review synthesis reuses the stored OpenAI-compatible client."""

    def test_build_reviewer_returns_the_shared_chat_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLAS_REVIEW_API_KEY", "value")
        reviewer = build_reviewer(provider())
        assert isinstance(reviewer, ChatContextualizer)
        reviewer.close()

    def test_missing_credential_is_an_authentication_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ATLAS_REVIEW_API_KEY", raising=False)
        with pytest.raises(AtlasReviewError) as caught:
            build_reviewer(provider())
        assert caught.value.stage == "llm_authentication"
        assert isinstance(caught.value.__cause__, MissingProviderCredentialError)
