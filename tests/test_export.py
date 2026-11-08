"""Markdown export: self-contained, sanitized, and safe to overwrite."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from atlas_review.contracts import EvidencePacket, ReviewFinding
from atlas_review.errors import AtlasReviewError
from atlas_review.export import (
    DRAFT_NOTICE,
    default_filename,
    export_markdown,
    render_markdown,
)
from atlas_review.storage.reviews import StoredReview
from packets import CASE, EXERCISE, MITIGATION, TECHNIQUE, finding, valid_report


def stored(base: StoredReview, **changes: Any) -> StoredReview:
    """Return the fixture review with explicit replacements."""
    return replace(base, **changes)


def with_finding(base: StoredReview, **changes: Any) -> StoredReview:
    """Return the fixture review whose only finding has explicit changes."""
    report = valid_report(finding(**changes))
    return replace(base, report=report)


class TestReportDocument:
    """The document carries the report, its citations, and the brief."""

    def test_document_states_it_is_a_draft(self, stored_review: StoredReview) -> None:
        markdown = render_markdown(stored_review)
        assert DRAFT_NOTICE in markdown
        assert markdown.startswith("# Internal knowledge assistant design review")

    @pytest.mark.parametrize(
        "expected",
        [
            "## Summary",
            "## Findings",
            "### F001: Untrusted retrieved instructions",
            "- Priority: review_next",
            "**Applicability.**",
            "**Source facts from the brief**",
            "**Assumptions and unknowns**",
            "**Controls**",
            "## Citations",
            "## Application brief",
            "## Evidence appendix",
            "Apache License 2.0",
            "2026.09",
            "E001",
            "E006",
        ],
        ids=[
            "summary",
            "findings",
            "finding-title",
            "priority",
            "applicability",
            "facts",
            "unknowns",
            "controls",
            "citations",
            "brief",
            "appendix",
            "attribution",
            "collection",
            "first-citation",
            "case-citation",
        ],
    )
    def test_document_contains_the_required_sections(
        self, stored_review: StoredReview, expected: str
    ) -> None:
        assert expected in render_markdown(stored_review)

    def test_brief_values_are_exported(self, stored_review: StoredReview) -> None:
        markdown = render_markdown(stored_review)
        assert "| Internal document repository; Workspace messages |" in markdown
        assert "Whether returned links are validated outside the model" in markdown

    def test_generated_context_is_not_exported(
        self, stored_review: StoredReview
    ) -> None:
        markdown = render_markdown(stored_review)
        assert "Context (generated)" not in markdown
        assert "Indirect prompt injection places instructions" in markdown
        assert "Input Validation: validate untrusted text" in markdown

    def test_credentials_are_not_exported(self, stored_review: StoredReview) -> None:
        markdown = render_markdown(stored_review)
        assert stored_review.config.llm.api_key_env is not None
        assert stored_review.config.llm.api_key_env not in markdown
        assert "api_key" not in markdown.lower()

    def test_review_without_a_report_says_so(self, stored_review: StoredReview) -> None:
        pending = stored(stored_review, report=None)
        markdown = render_markdown(pending)
        assert "No validated report is stored" in markdown
        assert "## Findings" not in markdown

    def test_empty_findings_report_is_explained(
        self, stored_review: StoredReview
    ) -> None:
        empty = valid_report(findings=[])
        markdown = render_markdown(stored(stored_review, report=empty))
        assert "No findings" in markdown


class TestCaseAndAssociationLabels:
    """Stored classifications and relationship kinds survive the export."""

    @pytest.mark.parametrize(
        ("entity_id", "route"),
        [
            pytest.param(
                TECHNIQUE,
                f"https://atlas.mitre.org/techniques/{TECHNIQUE}",
                id="technique",
            ),
            pytest.param(
                MITIGATION,
                f"https://atlas.mitre.org/mitigations/{MITIGATION}",
                id="mitigation",
            ),
            pytest.param(
                CASE,
                f"https://atlas.mitre.org/studies/{CASE}",
                id="case-study",
            ),
        ],
    )
    def test_citation_index_links_the_canonical_atlas_page(
        self, stored_review: StoredReview, entity_id: str, route: str
    ) -> None:
        markdown = render_markdown(stored_review)
        assert entity_id in markdown
        assert route in markdown

    def test_retained_source_locator_accompanies_each_link(
        self, stored_review: StoredReview
    ) -> None:
        markdown = render_markdown(stored_review)
        for item in stored_review.evidence.items:
            assert item.source_locator in markdown

    def test_case_study_keeps_its_stored_classification(
        self, stored_review: StoredReview
    ) -> None:
        assert f"{CASE} ({EXERCISE})" in render_markdown(stored_review)

    def test_parent_linked_mitigation_is_labelled(
        self, stored_review: StoredReview
    ) -> None:
        markdown = render_markdown(stored_review)
        assert f"{MITIGATION} (parent-linked via AML.T0051)" in markdown

    def test_direct_association_is_labelled_direct(
        self, stored_review: StoredReview
    ) -> None:
        packet = stored_review.evidence
        direct = EvidencePacket.model_validate(
            {
                **packet.model_dump(mode="json"),
                "allowed_associations": [
                    {
                        **packet.allowed_associations[1].model_dump(mode="json"),
                        "parent_technique_id": None,
                    }
                ],
            },
            strict=True,
        )
        markdown = render_markdown(replace(stored_review, evidence=direct))
        assert f"{MITIGATION} (direct)" in markdown

    def test_suggested_check_without_mitigation_is_labelled(
        self, stored_review: StoredReview
    ) -> None:
        control = {
            "action": "Add an independent output check.",
            "rationale": "The brief does not describe one.",
            "verification": "Identify who validates output links.",
            "mitigation_ids": [],
            "evidence_ids": ["E001"],
        }
        markdown = render_markdown(
            stored(stored_review, report=valid_report(finding(controls=[control])))
        )
        assert "not an ATLAS-prescribed control" in markdown


class TestSanitization:
    """Untrusted report text cannot break the document structure."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("escape\x1b[31mred", "escapered", id="escape"),
            pytest.param("bell\x07", "bell", id="control"),
            pytest.param("a | b", "a | b", id="pipe"),
            pytest.param("line\nbreak", "line break", id="newline"),
        ],
    )
    def test_untrusted_text_is_cleaned(
        self, stored_review: StoredReview, value: str, expected: str
    ) -> None:
        markdown = render_markdown(with_finding(stored_review, title=value))
        assert expected in markdown

    def test_appendix_rows_stay_single_line(self, stored_review: StoredReview) -> None:
        for line in render_markdown(stored_review).splitlines():
            assert "\x1b" not in line


class TestExportFiles:
    """Exports are written atomically and never clobber silently."""

    def test_export_writes_the_rendered_document(
        self, stored_review: StoredReview, tmp_path: Path
    ) -> None:
        target = export_markdown(stored_review, tmp_path)
        assert target.parent == tmp_path
        assert target.name == default_filename(stored_review)
        assert target.read_text(encoding="utf-8") == render_markdown(stored_review)
        assert not list(tmp_path.glob(".*tmp"))

    def test_second_export_uses_a_free_name(
        self, stored_review: StoredReview, tmp_path: Path
    ) -> None:
        first = export_markdown(stored_review, tmp_path)
        second = export_markdown(stored_review, tmp_path)
        assert second != first
        assert second.exists()
        assert first.exists()

    def test_overwrite_replaces_the_existing_file(
        self, stored_review: StoredReview, tmp_path: Path
    ) -> None:
        first = export_markdown(stored_review, tmp_path)
        first.write_text("stale", encoding="utf-8")
        again = export_markdown(stored_review, tmp_path, overwrite=True)
        assert again == first
        assert again.read_text(encoding="utf-8") == render_markdown(stored_review)

    def test_unusable_destination_is_an_export_error(
        self, stored_review: StoredReview, tmp_path: Path
    ) -> None:
        blocked = tmp_path / "file"
        blocked.write_text("not a directory", encoding="utf-8")
        with pytest.raises(AtlasReviewError) as caught:
            export_markdown(stored_review, blocked)
        assert caught.value.code == "export_failed"
        assert caught.value.stage == "export"


def test_finding_helper_matches_the_contract() -> None:
    finding_record: ReviewFinding = valid_report().findings[0]
    assert finding_record.finding_id == "F001"
