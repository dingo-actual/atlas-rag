"""Review-screen rendering: evidence, request summary, report, and settings."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from rich.console import Console, RenderableType

from atlas_review.config import LLMConfig
from atlas_review.contracts import EvidenceItem, GraphHop, GraphPath, ReviewAttempt
from atlas_review.storage.reviews import ReviewSummary, StoredReview
from atlas_review.tui.review_render import (
    render_configuration,
    render_evidence_detail,
    render_evidence_packet,
    render_generated_context,
    render_request_summary,
    render_review,
    render_review_header,
    render_reviews,
)
from packets import (
    CASE,
    EXERCISE,
    MITIGATION,
    PARENT,
    TECHNIQUE,
    context_entry,
    default_packet,
    digest,
    generated_passage,
)


def text(renderable: RenderableType) -> str:
    """Render one Rich renderable to plain text without colors."""
    console = Console(record=True, width=140, color_system=None)
    console.print(renderable)
    return console.export_text()


def provider(**overrides: object) -> LLMConfig:
    """Return a provider configuration with explicit overrides."""
    base = LLMConfig(
        base_url="https://provider.example/v1", model="fake-model-1"
    ).model_dump(mode="json")
    return LLMConfig.model_validate({**base, **overrides}, strict=True)


class TestEvidencePacket:
    """The packet view lists items and warns about incomplete coverage."""

    def test_items_and_case_classification_are_listed(self) -> None:
        rendered = text(render_evidence_packet(default_packet()))
        assert "E001" in rendered
        assert CASE in rendered
        assert EXERCISE in rendered

    def test_generated_context_is_called_out(self) -> None:
        rendered = text(render_evidence_packet(default_packet()))
        assert "Generated retrieval context is excluded" in rendered

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"missing_categories": ("mitigation",)}, id="missing"),
            pytest.param({"exclusions": ("AML.T0051::case study",)}, id="excluded"),
        ],
    )
    def test_coverage_notices_are_shown(self, kwargs: dict[str, Any]) -> None:
        rendered = text(render_evidence_packet(default_packet(), **kwargs))
        assert "Warning:" in rendered or "Excluded from the packet:" in rendered

    def test_item_detail_shows_text_and_provenance(self) -> None:
        item = default_packet().items[2]
        rendered = text(render_evidence_detail(item))
        assert "Input Validation" in rendered
        assert item.source_locator in rendered

    def test_item_detail_shows_the_traversed_graph_path(self) -> None:
        item = EvidenceItem(
            evidence_id="E008",
            item_kind="passage",
            entity_ids=[TECHNIQUE],
            passage_id=9,
            text="Indirect prompt injection detail.",
            source_locator=f"/techniques/{TECHNIQUE}/description",
            collection_version="2026.09",
            retrieval_channels=["graph"],
            graph_path=GraphPath(
                seed_entity_id=TECHNIQUE,
                hops=[
                    GraphHop(
                        edge_id=digest("employs"),
                        source_id=CASE,
                        target_id=TECHNIQUE,
                        relation="employs",
                        traversed_inverse=True,
                    )
                ],
                entity_id=CASE,
            ),
        )
        rendered = text(render_evidence_detail(item))
        assert "employs" in rendered
        assert CASE in rendered
        assert "inverse" in rendered

    def test_item_without_a_path_omits_the_path_table(self) -> None:
        rendered = text(render_evidence_detail(default_packet().items[2]))
        assert "Graph path" not in rendered


class TestGeneratedContextInspection:
    """The generated note is shown as a retrieval aid, not as evidence."""

    def test_note_is_labelled_as_generated_and_shown_with_the_excerpt(self) -> None:
        passage = generated_passage()
        rendered = text(render_generated_context(passage, context_entry()))
        assert "LLM-generated retrieval aid" in rendered
        assert passage.context_text in rendered
        assert passage.source_text in rendered
        assert "chunk-context-v1" in rendered

    def test_support_locators_and_cache_key_are_shown(self) -> None:
        passage = generated_passage()
        entry = context_entry()
        rendered = text(render_generated_context(passage, entry))
        assert entry.support_locators[0] in rendered
        assert entry.cache_key[:12] in rendered

    def test_missing_cache_entry_still_shows_the_stored_note(self) -> None:
        passage = generated_passage()
        rendered = text(render_generated_context(passage, None))
        assert passage.context_text in rendered
        assert "LLM-generated retrieval aid" in rendered


class TestRequestSummary:
    """The confirmation shows where the request goes and what it carries."""

    def test_destination_and_model_are_shown_without_credentials(self) -> None:
        rendered = text(render_request_summary(provider(), 7))
        assert "provider.example" in rendered
        assert "fake-model-1" in rendered
        assert "7" in rendered

    def test_host_only_is_shown_for_the_endpoint(self) -> None:
        rendered = text(render_request_summary(provider(), 7))
        assert "/v1" not in rendered

    def test_sending_is_described(self) -> None:
        rendered = text(render_request_summary(provider(), 7))
        assert "configured endpoint" in rendered


class TestReviewHeader:
    """The evidence screen names the review and its retrieval focus."""

    def test_title_and_focus_are_shown(self) -> None:
        rendered = text(
            render_review_header(
                "Internal assistant review", "indirect prompt injection"
            )
        )
        assert "Internal assistant review" in rendered
        assert "indirect prompt injection" in rendered

    def test_untrusted_text_is_sanitized(self) -> None:
        rendered = text(render_review_header("title\x1b[31m", "focus\x07"))
        assert "\x1b" not in rendered
        assert "title" in rendered


class TestReviewRendering:
    """A stored report renders its priorities, cases, and caveats."""

    def test_report_shows_priority_and_citations(
        self, stored_review: StoredReview
    ) -> None:
        rendered = text(render_review(stored_review))
        assert "review_next" in rendered
        assert "E001" in rendered
        assert "F001" in rendered

    def test_case_classification_is_preserved(
        self, stored_review: StoredReview
    ) -> None:
        assert f"{CASE} ({EXERCISE})" in text(render_review(stored_review))

    def test_mitigation_association_is_caveated(
        self, stored_review: StoredReview
    ) -> None:
        rendered = text(render_review(stored_review))
        assert MITIGATION in rendered
        assert "not evidence of effectiveness" in rendered

    def test_parent_linked_mitigation_is_labelled_in_the_terminal(
        self, stored_review: StoredReview
    ) -> None:
        rendered = text(render_review(stored_review))
        assert f"{MITIGATION} (parent-linked via {PARENT})" in rendered

    def test_scan_disclaimer_is_present(self, stored_review: StoredReview) -> None:
        assert "not a scan or test result" in text(render_review(stored_review))

    def test_retrieved_review_without_a_report_explains_the_state(
        self, stored_review: StoredReview
    ) -> None:
        rendered = text(render_review(replace(stored_review, report=None)))
        assert "No validated report is stored" in rendered

    def test_attempts_are_listed(self, stored_review: StoredReview) -> None:
        attempt = ReviewAttempt.model_validate(
            {
                "schema_version": 1,
                "review_id": stored_review.review_id,
                "attempt_number": 1,
                "state": "failed",
                "started_at": "2026-11-07T12:00:00+00:00",
                "provider_name": "custom-compatible",
                "base_url": "https://provider.example/v1",
                "model": "fake-model-1",
                "output_token_parameter": "max_tokens",
                "output_tokens": 2500,
                "response_format_mode": "text",
                "max_findings": 5,
                "request_characters": 9000,
                "evidence_item_count": 7,
                "error_code": "invalid_report_json",
            },
            strict=True,
        )
        rendered = text(render_review(stored_review, [attempt]))
        assert "invalid_report_json" in rendered

    def test_attempt_without_reported_output_tokens_is_not_shown_as_none(
        self, stored_review: StoredReview
    ) -> None:
        attempt = ReviewAttempt.model_validate(
            {
                "schema_version": 1,
                "review_id": stored_review.review_id,
                "attempt_number": 1,
                "state": "complete",
                "started_at": "2026-11-07T12:00:00+00:00",
                "provider_name": "custom-compatible",
                "base_url": "https://provider.example/v1",
                "model": "fake-model-1",
                "output_token_parameter": "max_tokens",
                "output_tokens": 2500,
                "response_format_mode": "text",
                "max_findings": 5,
                "request_characters": 9000,
                "evidence_item_count": 7,
                "response_text": '{"schema_version": 1, "summary": "x"}',
                "usage_input_tokens": 1200,
            },
            strict=True,
        )
        rendered = text(render_review(stored_review, [attempt]))
        assert "1200/-" in rendered
        assert "None" not in rendered


class TestReviewsList:
    """The saved-review list shows identifiers, state, and collection."""

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_list_handles_empty_and_populated(self, count: int) -> None:
        summaries = [
            ReviewSummary(
                review_id=f"review-{index}",
                created_at="2026-11-07T12:00:00+00:00",
                title="Internal assistant review",
                state="complete",
                collection_version="2026.09",
            )
            for index in range(count)
        ]
        rendered = text(render_reviews(summaries))
        assert ("No saved reviews yet." in rendered) is (count == 0)


class TestConfigurationRendering:
    """Provider settings render names and presence, never secret values."""

    @pytest.fixture
    def secret(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
        """Set a credential variable for the duration of one test."""
        value = "sk-do-not-display"
        monkeypatch.setenv("ATLAS_REVIEW_API_KEY", value)
        yield value
        monkeypatch.delenv("ATLAS_REVIEW_API_KEY", raising=False)

    def test_secret_value_is_never_rendered(self, secret: str) -> None:
        rendered = text(render_configuration(provider(), credential_present=True))
        assert secret not in rendered
        assert "ATLAS_REVIEW_API_KEY" in rendered

    def test_presence_is_reported_not_the_value(self) -> None:
        config = provider()
        assert "yes" in text(render_configuration(config, credential_present=True))
        assert "no" in text(render_configuration(config, credential_present=False))

    def test_environment_name_is_the_documented_default(self) -> None:
        assert os.environ.get("ATLAS_REVIEW_API_KEY") is None
        assert "ATLAS_REVIEW_API_KEY" in text(
            render_configuration(provider(), credential_present=False)
        )
