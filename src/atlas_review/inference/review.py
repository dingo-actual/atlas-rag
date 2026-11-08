"""One explicit review-synthesis attempt: request, validate, record.

A Generate action makes exactly one provider request. The started attempt is
stored before the request; its outcome is stored after it, so a failure keeps
the brief, the frozen evidence, and any earlier accepted report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from atlas_review.config import LLMConfig
from atlas_review.contracts import (
    EvidencePacket,
    ReviewAttempt,
    ReviewReport,
    ReviewRequest,
)
from atlas_review.errors import AtlasReviewError
from atlas_review.inference.client import ProviderReply
from atlas_review.inference.prompt import ReviewPrompt, build_prompt
from atlas_review.inference.validate import decode_report, validate_report


class Reviewer(Protocol):
    """Answer one review-synthesis request."""

    def __call__(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        output_tokens: int,
        temperature: float | None,
    ) -> ProviderReply:
        """Return the provider reply for one review request."""
        ...


class SynthesisStore(Protocol):
    """Persist one review's attempts and its accepted report."""

    def record(self, attempt: ReviewAttempt) -> None:
        """Store or replace one numbered attempt."""
        ...

    def complete(self, report: ReviewReport) -> None:
        """Store the accepted report."""
        ...

    def fail(self) -> None:
        """Mark the review failed while it has no accepted report."""
        ...


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """Result of one accepted synthesis attempt.

    Instance Attributes:
        review_id (str): Review the attempt belongs to.
        attempt (ReviewAttempt): Finished attempt with its reported usage.
        report (ReviewReport): Validated report stored with the review.
    """

    review_id: str
    attempt: ReviewAttempt
    report: ReviewReport


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """Identity of the attempt about to be recorded.

    Instance Attributes:
        review_id (str): Review the attempt belongs to.
        attempt_number (int): Numbered attempt for this review.
        started_at (str): ISO 8601 start timestamp of the attempt.
    """

    review_id: str
    attempt_number: int
    started_at: str


def generate_review(
    brief: ReviewRequest,
    packet: EvidencePacket,
    config: LLMConfig,
    *,
    plan: AttemptPlan,
    reviewer: Reviewer,
    store: SynthesisStore,
) -> ReviewOutcome:
    """Make one provider request and store its validated outcome.

    Args:
        brief (ReviewRequest): Validated application brief.
        packet (EvidencePacket): Frozen evidence packet for this review.
        config (LLMConfig): Provider settings and output allowance.
        plan (AttemptPlan): Review identifier, attempt number, and start time.
        reviewer (Reviewer): Provider adapter.
        store (SynthesisStore): Persistence for attempts and the report.

    Returns:
        (ReviewOutcome) Finished attempt and validated report.

    Raises:
        AtlasReviewError: The request exceeds its budget, the provider fails,
            or the response cannot be validated against the packet.
    """
    prompt = build_prompt(brief, packet, config)
    started = _attempt(prompt, config, plan)
    store.record(started)
    try:
        reply = reviewer(
            prompt.system,
            prompt.user,
            output_tokens=config.output_tokens,
            temperature=config.temperature,
        )
    except AtlasReviewError as error:
        store.record(_failed(started, error, reply=None))
        store.fail()
        raise
    except KeyboardInterrupt:
        store.record(_replace(started, state="interrupted"))
        raise
    answered = _answered(started, reply)
    try:
        report = decode_report(reply.text)
        validate_report(report, prompt.packet, max_findings=config.max_findings)
    except AtlasReviewError as error:
        store.record(_failed(answered, error, reply=reply))
        store.fail()
        raise
    finished = _completed(answered)
    store.record(finished)
    store.complete(report)
    return ReviewOutcome(review_id=plan.review_id, attempt=finished, report=report)


def _attempt(
    prompt: ReviewPrompt, config: LLMConfig, plan: AttemptPlan
) -> ReviewAttempt:
    """Build the started attempt recorded before the provider request."""
    return ReviewAttempt(
        schema_version=1,
        review_id=plan.review_id,
        attempt_number=plan.attempt_number,
        state="started",
        started_at=plan.started_at,
        provider_name=config.provider_name,
        base_url=config.base_url,
        model=config.model,
        output_token_parameter=config.output_token_parameter,
        output_tokens=config.output_tokens,
        temperature=config.temperature,
        response_format_mode=config.response_format_mode,
        max_findings=config.max_findings,
        request_characters=len(prompt.system) + len(prompt.user),
        evidence_item_count=len(prompt.packet.items),
    )


def _answered(attempt: ReviewAttempt, reply: ProviderReply) -> ReviewAttempt:
    """Add the sanitized provider reply metadata to an attempt."""
    return _replace(
        attempt,
        response_text=reply.text,
        response_id=reply.response_id,
        response_model=reply.model,
        usage_input_tokens=reply.input_tokens,
        usage_output_tokens=reply.output_tokens,
        duration_seconds=reply.duration_seconds,
    )


def _completed(attempt: ReviewAttempt) -> ReviewAttempt:
    """Mark one answered attempt complete."""
    return _replace(attempt, state="complete")


def _failed(
    attempt: ReviewAttempt, error: AtlasReviewError, *, reply: ProviderReply | None
) -> ReviewAttempt:
    """Record a failed attempt with its sanitized error and response text."""
    changes: dict[str, object] = {
        "state": "failed",
        "error_code": error.code,
        "error_message": error.message,
        "validation_details": list(error.details),
    }
    if reply is not None:
        changes["duration_seconds"] = reply.duration_seconds
        changes["response_text"] = reply.text
    return _replace(attempt, **changes)


def _replace(attempt: ReviewAttempt, **changes: object) -> ReviewAttempt:
    """Validate a replacement attempt instead of copying it unchecked."""
    return ReviewAttempt.model_validate(
        {**attempt.model_dump(mode="json"), **changes}, strict=True
    )
