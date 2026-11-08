"""Review persistence: retrieved snapshots, attempts, completion, and reopening."""

from __future__ import annotations

from typing import Any

import pytest

from atlas_review.config import AppConfig
from atlas_review.contracts import ReviewAttempt, ReviewReport, ReviewRequest
from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.storage.reviews import ReviewRetrieval
from packets import CREATED_AT, default_packet, sample_trace

REVIEW_ID = "review-1"


def report() -> ReviewReport:
    """Return a minimal valid report."""
    return ReviewReport.model_validate(
        {"schema_version": 1, "summary": "insufficient evidence"}, strict=True
    )


def attempt(number: int, **overrides: Any) -> ReviewAttempt:
    """Return one attempt record with explicit overrides."""
    body: dict[str, Any] = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "attempt_number": number,
        "state": "started",
        "started_at": CREATED_AT,
        "provider_name": "custom-compatible",
        "base_url": "https://provider.example/v1",
        "model": "fake-model-1",
        "output_token_parameter": "max_tokens",
        "output_tokens": 2500,
        "temperature": None,
        "response_format_mode": "text",
        "max_findings": 5,
        "request_characters": 9000,
        "evidence_item_count": 7,
        **overrides,
    }
    return ReviewAttempt.model_validate(body, strict=True)


@pytest.fixture
def retrieved(
    stored_session: WorkspaceSession, review_brief: ReviewRequest
) -> WorkspaceSession:
    """Store one retrieved review row over the fixture generation."""
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
    return stored_session


class TestRetrievedReview:
    """A retrieved run is saved before any provider request."""

    def test_retrieved_snapshot_reopens_without_a_report(
        self, retrieved: WorkspaceSession
    ) -> None:
        stored = retrieved.reviews.review(REVIEW_ID)
        assert stored.state == "retrieved"
        assert stored.report is None
        assert stored.evidence == default_packet()
        assert stored.request.title == "Internal knowledge assistant design review"
        assert stored.config == AppConfig.model_validate(
            retrieved.config.model_dump(mode="json"), strict=True
        )
        assert stored.trace.generation_id == stored.generation_id

    def test_unknown_review_is_reported(self, session: WorkspaceSession) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            session.reviews.review("missing-review")
        assert caught.value.code == "unknown_review"

    def test_summaries_list_the_retrieved_review(
        self, retrieved: WorkspaceSession
    ) -> None:
        summaries = retrieved.reviews.summaries()
        assert [summary.review_id for summary in summaries] == [REVIEW_ID]
        assert summaries[0].state == "retrieved"
        assert summaries[0].collection_version == "2026.09"


class TestAttempts:
    """Attempts are numbered, replaceable, and readable in order."""

    def test_attempt_numbers_start_at_one(self, retrieved: WorkspaceSession) -> None:
        assert retrieved.reviews.next_attempt_number(REVIEW_ID) == 1

    def test_started_attempt_is_replaced_by_its_outcome(
        self, retrieved: WorkspaceSession
    ) -> None:
        retrieved.reviews.record(attempt(1))
        assert retrieved.reviews.next_attempt_number(REVIEW_ID) == 2
        retrieved.reviews.record(
            attempt(1, state="failed", error_code="invalid_report_json")
        )
        stored = retrieved.reviews.attempts(REVIEW_ID)
        assert len(stored) == 1
        assert stored[0].state == "failed"
        assert stored[0].error_code == "invalid_report_json"

    def test_attempts_keep_their_request_order(
        self, retrieved: WorkspaceSession
    ) -> None:
        retrieved.reviews.record(attempt(2, response_text="later"))
        retrieved.reviews.record(attempt(1, response_text="earlier"))
        assert [
            item.attempt_number for item in retrieved.reviews.attempts(REVIEW_ID)
        ] == [
            1,
            2,
        ]


class TestCompletion:
    """A report marks the review complete and survives a failed retry."""

    def test_completion_stores_the_report(self, retrieved: WorkspaceSession) -> None:
        retrieved.reviews.complete(REVIEW_ID, report())
        stored = retrieved.reviews.review(REVIEW_ID)
        assert stored.state == "complete"
        assert stored.report == report()

    def test_failure_before_success_marks_the_review_failed(
        self, retrieved: WorkspaceSession
    ) -> None:
        retrieved.reviews.record_failure(REVIEW_ID)
        assert retrieved.reviews.review(REVIEW_ID).state == "failed"

    def test_failure_after_success_keeps_the_review_complete(
        self, retrieved: WorkspaceSession
    ) -> None:
        retrieved.reviews.complete(REVIEW_ID, report())
        retrieved.reviews.record_failure(REVIEW_ID)
        stored = retrieved.reviews.review(REVIEW_ID)
        assert stored.state == "complete"
        assert stored.report == report()
