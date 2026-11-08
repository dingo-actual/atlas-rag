"""Saved reviews and their numbered provider attempts.

A retrieved run is stored before any provider request. Each Generate action
saves a `started` attempt and then its outcome, so a failed attempt keeps the
brief, the frozen evidence, and the earlier accepted report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from atlas_review.config import AppConfig
from atlas_review.contracts import (
    EvidencePacket,
    RetrievalTrace,
    ReviewAttempt,
    ReviewReport,
    ReviewRequest,
)
from atlas_review.errors import STAGE_STORAGE, AtlasReviewError
from atlas_review.storage.codec import decode_model, encode_model
from atlas_review.storage.database import WorkspaceDatabase

ReviewState = Literal["retrieved", "complete", "failed"]

_INSERT_REVIEW = """
INSERT INTO reviews (
    review_id, generation_id, created_at, title, state,
    request_msgpack, config_msgpack, trace_msgpack, evidence_msgpack,
    report_msgpack
) VALUES (?, ?, ?, ?, 'retrieved', ?, ?, ?, ?, NULL)
"""
_UPSERT_ATTEMPT = """
INSERT INTO review_attempts (
    review_id, attempt_number, started_at, state, payload_msgpack
) VALUES (?, ?, ?, ?, ?)
ON CONFLICT (review_id, attempt_number) DO UPDATE SET
    state = excluded.state,
    payload_msgpack = excluded.payload_msgpack
"""
_STARTED_ATTEMPTS = "SELECT review_id, attempt_number, payload_msgpack FROM review_attempts WHERE state = 'started'"
_MARK_INTERRUPTED = """
UPDATE review_attempts SET state = 'interrupted', payload_msgpack = ?
WHERE review_id = ? AND attempt_number = ?
"""


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    """List entry for one saved review.

    Instance Attributes:
        review_id (str): Application-generated review identifier.
        created_at (str): ISO 8601 retrieval timestamp.
        title (str): Brief title recorded with the review.
        state (ReviewState): ``retrieved``, ``complete``, or ``failed``.
        collection_version (str): ATLAS collection version of the evidence.
    """

    review_id: str
    created_at: str
    title: str
    state: ReviewState
    collection_version: str


@dataclass(frozen=True, slots=True)
class StoredReview:
    """One saved review with its frozen evidence and accepted report.

    Instance Attributes:
        review_id (str): Application-generated review identifier.
        generation_id (str): Corpus generation the evidence came from.
        created_at (str): ISO 8601 retrieval timestamp.
        title (str): Brief title recorded with the review.
        state (ReviewState): ``retrieved``, ``complete``, or ``failed``.
        request (ReviewRequest): Validated brief used for retrieval.
        config (AppConfig): Configuration snapshot taken at retrieval time.
        trace (RetrievalTrace): Retrieval trace saved with the review.
        evidence (EvidencePacket): Frozen evidence packet.
        report (ReviewReport | None): Accepted report, once generated.
    """

    review_id: str
    generation_id: str
    created_at: str
    title: str
    state: ReviewState
    request: ReviewRequest
    config: AppConfig
    trace: RetrievalTrace
    evidence: EvidencePacket
    report: ReviewReport | None


@dataclass(frozen=True, slots=True)
class ReviewRetrieval:
    """Inputs frozen when a review is retrieved, before any provider call.

    Instance Attributes:
        review_id (str): Application-generated review identifier.
        generation_id (str): Generation the evidence came from.
        created_at (str): ISO 8601 retrieval timestamp.
        request (ReviewRequest): Validated brief used for retrieval.
        config (AppConfig): Configuration snapshot without any secret.
        trace (RetrievalTrace): Validated retrieval trace.
        evidence (EvidencePacket): Frozen evidence packet.
    """

    review_id: str
    generation_id: str
    created_at: str
    request: ReviewRequest
    config: AppConfig
    trace: RetrievalTrace
    evidence: EvidencePacket


@dataclass(frozen=True, slots=True)
class ReviewWriter:
    """Persist one review's attempts and its accepted report."""

    repository: ReviewRepository
    review_id: str

    def record(self, attempt: ReviewAttempt) -> None:
        """Store or replace one numbered attempt."""
        self.repository.record(attempt)

    def complete(self, report: ReviewReport) -> None:
        """Store the accepted report and mark the review complete."""
        self.repository.complete(self.review_id, report)

    def fail(self) -> None:
        """Mark the review failed when it has no accepted report."""
        self.repository.record_failure(self.review_id)


class ReviewRepository:
    """Read and write saved reviews over one workspace database.

    Instance Attributes:
        _database (WorkspaceDatabase): Workspace connection factory.
    """

    def __init__(self, database: WorkspaceDatabase) -> None:
        """Bind the repository to one workspace database.

        Args:
            database (WorkspaceDatabase): Workspace connection factory.
        """
        self._database = database

    def save_retrieved(self, retrieval: ReviewRetrieval) -> None:
        """Store one retrieved run before any provider request.

        Args:
            retrieval (ReviewRetrieval): Frozen brief, evidence, and identity.
        """
        with self._database.transaction() as connection:
            connection.execute(
                _INSERT_REVIEW,
                (
                    retrieval.review_id,
                    retrieval.generation_id,
                    retrieval.created_at,
                    retrieval.request.title,
                    encode_model(retrieval.request),
                    encode_model(retrieval.config),
                    encode_model(retrieval.trace),
                    encode_model(retrieval.evidence),
                ),
            )

    def record(self, attempt: ReviewAttempt) -> None:
        """Store or replace one attempt of its review."""
        with self._database.transaction() as connection:
            connection.execute(
                _UPSERT_ATTEMPT,
                (
                    attempt.review_id,
                    attempt.attempt_number,
                    attempt.started_at,
                    attempt.state,
                    encode_model(attempt),
                ),
            )

    def interrupt_started(self) -> int:
        """Mark attempts a previous process left `started` as interrupted.

        A `started` row with no outcome means the process died or was
        cancelled mid-request. It is never resubmitted automatically.

        Returns:
            (int) Number of attempts marked interrupted.
        """
        rows = self._database.connect().execute(_STARTED_ATTEMPTS).fetchall()
        if not rows:
            return 0
        with self._database.transaction() as connection:
            for row in rows:
                attempt = decode_model(row["payload_msgpack"], ReviewAttempt)
                interrupted = ReviewAttempt.model_validate(
                    {**attempt.model_dump(mode="json"), "state": "interrupted"},
                    strict=True,
                )
                connection.execute(
                    _MARK_INTERRUPTED,
                    (
                        encode_model(interrupted),
                        row["review_id"],
                        row["attempt_number"],
                    ),
                )
        return len(rows)

    def complete(self, review_id: str, report: ReviewReport) -> None:
        """Store the accepted report and mark the review complete."""
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE reviews SET state = 'complete', report_msgpack = ? "
                "WHERE review_id = ?",
                (encode_model(report), review_id),
            )

    def record_failure(self, review_id: str) -> None:
        """Mark a review failed while it still has no accepted report."""
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE reviews SET state = 'failed' "
                "WHERE review_id = ? AND report_msgpack IS NULL",
                (review_id,),
            )

    def next_attempt_number(self, review_id: str) -> int:
        """Return the next attempt number for one review."""
        row = (
            self._database.connect()
            .execute(
                "SELECT coalesce(max(attempt_number), 0) FROM review_attempts "
                "WHERE review_id = ?",
                (review_id,),
            )
            .fetchone()
        )
        return int(row[0]) + 1

    def attempts(self, review_id: str) -> tuple[ReviewAttempt, ...]:
        """Return every recorded attempt of one review in request order."""
        rows = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM review_attempts "
                "WHERE review_id = ? ORDER BY attempt_number",
                (review_id,),
            )
            .fetchall()
        )
        return tuple(
            decode_model(row["payload_msgpack"], ReviewAttempt) for row in rows
        )

    def summaries(self) -> tuple[ReviewSummary, ...]:
        """Return saved reviews newest first."""
        rows = (
            self._database.connect()
            .execute(
                """
            SELECT r.review_id, r.created_at, r.title, r.state,
                   g.collection_version
            FROM reviews r
            JOIN generations g ON g.generation_id = r.generation_id
            ORDER BY r.rowid DESC
            """
            )
            .fetchall()
        )
        return tuple(
            ReviewSummary(
                review_id=row["review_id"],
                created_at=row["created_at"],
                title=row["title"],
                state=row["state"],
                collection_version=row["collection_version"],
            )
            for row in rows
        )

    def review(self, review_id: str) -> StoredReview:
        """Decode and revalidate one saved review.

        Args:
            review_id (str): Stored review identifier.

        Returns:
            (StoredReview) Saved inputs, frozen evidence, and optional report.

        Raises:
            AtlasReviewError: The review is not stored.
        """
        row = (
            self._database.connect()
            .execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,))
            .fetchone()
        )
        if row is None:
            raise AtlasReviewError(
                code="unknown_review",
                stage=STAGE_STORAGE,
                message=f"review {review_id} is not stored",
                recovery="List saved reviews and select a stored identifier.",
            )
        payload = row["report_msgpack"]
        return StoredReview(
            review_id=row["review_id"],
            generation_id=row["generation_id"],
            created_at=row["created_at"],
            title=row["title"],
            state=row["state"],
            request=decode_model(row["request_msgpack"], ReviewRequest),
            config=decode_model(row["config_msgpack"], AppConfig),
            trace=decode_model(row["trace_msgpack"], RetrievalTrace),
            evidence=decode_model(row["evidence_msgpack"], EvidencePacket),
            report=(None if payload is None else decode_model(payload, ReviewReport)),
        )
