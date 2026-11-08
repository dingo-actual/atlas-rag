"""Review-synthesis contract tests: attempts and report shape."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from atlas_review.contracts import ReviewAttempt, ReviewReport
from atlas_review.storage.codec import decode_model, encode_model

ATTEMPT: dict[str, Any] = {
    "schema_version": 1,
    "review_id": "review-1",
    "attempt_number": 1,
    "state": "started",
    "started_at": "2026-11-07T00:00:00+00:00",
    "provider_name": "custom-compatible",
    "base_url": "https://provider.example/v1",
    "model": "fake-model-1",
    "output_token_parameter": "max_tokens",
    "output_tokens": 2500,
    "temperature": None,
    "response_format_mode": "text",
    "max_findings": 5,
    "request_characters": 12000,
    "evidence_item_count": 12,
}

COMPLETE: dict[str, Any] = {
    **ATTEMPT,
    "state": "complete",
    "response_text": '{"schema_version": 1, "summary": "s"}',
    "response_id": "chatcmpl-1",
    "response_model": "fake-model-1",
    "usage_input_tokens": 900,
    "usage_output_tokens": 120,
    "duration_seconds": 1.5,
}


class TestReviewAttempt:
    """A recorded attempt keeps its state and outcome coherent."""

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(ATTEMPT, id="started"),
            pytest.param(COMPLETE, id="complete"),
            pytest.param(
                {**ATTEMPT, "state": "failed", "error_code": "provider_request_failed"},
                id="failed",
            ),
            pytest.param(
                {**ATTEMPT, "state": "interrupted", "response_text": "partial"},
                id="interrupted",
            ),
        ],
    )
    def test_documented_attempts_validate(self, record: dict[str, Any]) -> None:
        attempt = ReviewAttempt.model_validate(record, strict=True)
        assert attempt.review_id == "review-1"

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            pytest.param({"attempt_number": 0}, "attempt_number", id="zero-attempt"),
            pytest.param({"state": "retrying"}, "state", id="unknown-state"),
            pytest.param({"output_tokens": "2500"}, "output_tokens", id="string"),
            pytest.param({"extra": 1}, "extra", id="extra-field"),
            pytest.param({"state": "failed"}, "error code", id="failed-without-error"),
            pytest.param(
                {"state": "complete", "error_code": "boom"},
                "must not record an error code",
                id="complete-with-error",
            ),
            pytest.param({"state": "complete"}, "response text", id="complete-no-text"),
        ],
    )
    def test_invalid_attempts_are_rejected(
        self, changes: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            ReviewAttempt.model_validate({**ATTEMPT, **changes}, strict=True)

    def test_attempt_round_trips_through_the_msgspec_adapter(self) -> None:
        attempt = ReviewAttempt.model_validate(COMPLETE, strict=True)
        assert decode_model(encode_model(attempt), ReviewAttempt) == attempt


class TestReviewReport:
    """The report contract keeps the documented bounds."""

    @pytest.mark.parametrize(
        "changes",
        [
            pytest.param({"summary": ""}, id="empty-summary"),
            pytest.param({"unanswered_questions": ["q"] * 21}, id="too-many-questions"),
            pytest.param({"schema_version": True}, id="boolean-version"),
        ],
    )
    def test_unusable_reports_are_rejected(self, changes: dict[str, Any]) -> None:
        body = {"schema_version": 1, "summary": "s", "findings": [], **changes}
        with pytest.raises(ValidationError):
            ReviewReport.model_validate(body, strict=True)

    def test_report_allows_zero_findings(self) -> None:
        report = ReviewReport.model_validate(
            {"schema_version": 1, "summary": "insufficient evidence"}, strict=True
        )
        assert report.findings == []
