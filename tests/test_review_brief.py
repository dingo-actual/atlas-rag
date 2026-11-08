"""Brief-entry helpers used by the review screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from atlas_review.contracts import ReviewRequest
from atlas_review.errors import AtlasReviewError
from atlas_review.tui.review import BRIEF_FIELDS, blank_brief, save_brief

REQUIRED: dict[str, str] = {
    "title": "Internal assistant design review",
    "review_question": "What should we verify before rollout?",
    "retrieval_focus": "indirect prompt injection through retrieved content",
    "system_description": "An employee assistant answers from internal documents.",
}


class TestBlankBrief:
    """The editable draft stays inside the brief contract."""

    def test_draft_uses_only_contract_fields(self) -> None:
        assert set(blank_brief()) <= set(ReviewRequest.model_fields)

    def test_every_editable_field_is_a_contract_field(self) -> None:
        fields = {field for field, _ in BRIEF_FIELDS}
        assert fields <= set(ReviewRequest.model_fields)
        assert set(REQUIRED) <= fields
        assert "schema_version" not in fields

    def test_draft_validates_once_the_required_text_is_entered(self) -> None:
        draft: dict[str, Any] = {**blank_brief(), **REQUIRED}
        brief = ReviewRequest.model_validate(draft, strict=True)
        assert brief.title == REQUIRED["title"]
        assert brief.unknowns == []

    def test_blank_draft_is_not_valid_before_entry(self) -> None:
        draft = {**blank_brief(), **REQUIRED, "system_description": ""}
        with pytest.raises(ValidationError):
            ReviewRequest.model_validate(draft, strict=True)


class TestSavedBrief:
    """A brief can be written as JSON and loaded back unchanged."""

    @pytest.fixture
    def brief(self) -> ReviewRequest:
        """Return one valid brief."""
        return ReviewRequest.model_validate({**blank_brief(), **REQUIRED}, strict=True)

    def test_saved_brief_reloads_through_the_contract(
        self, brief: ReviewRequest, tmp_path: Path
    ) -> None:
        target = save_brief(brief, tmp_path / "drafts" / "brief.json")
        reloaded = ReviewRequest.model_validate_json(
            target.read_text(encoding="utf-8"), strict=True
        )
        assert reloaded == brief

    def test_unwritable_destination_is_reported(
        self, brief: ReviewRequest, tmp_path: Path
    ) -> None:
        blocked = tmp_path / "file"
        blocked.write_text("not a directory", encoding="utf-8")
        with pytest.raises(AtlasReviewError) as caught:
            save_brief(brief, blocked / "brief.json")
        assert caught.value.code == "export_failed"
