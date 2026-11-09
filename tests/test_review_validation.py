"""Report decoding and evidence/relationship validation of review output."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from atlas_review.errors import AtlasReviewError
from atlas_review.inference.validate import (
    MAX_REPORT_BYTES,
    decode_report,
    validate_report,
)
from packets import (
    CASE,
    CONTROL,
    EXERCISE,
    FINDING,
    MITIGATION,
    PARENT,
    TECHNIQUE,
    default_associations,
    default_items,
    default_packet,
    finding,
    packet,
    second_finding,
    valid_report,
)


class TestDecoding:
    """Provider text becomes a validated report only when it matches the contract."""

    def test_valid_json_becomes_a_report(self) -> None:
        text = valid_report().model_dump_json()
        assert decode_report(text) == valid_report()

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            pytest.param("not json", "invalid_report_json", id="not-json"),
            pytest.param('{"schema_version": 1}', "invalid_report_json", id="missing"),
            pytest.param('{"summary": "x"}', "invalid_report_json", id="no-version"),
            pytest.param(
                '{"schema_version": 1, "summary": "x", "extra": 1}',
                "invalid_report_json",
                id="extra-field",
            ),
            pytest.param(
                '{"schema_version": "1", "summary": "x"}',
                "invalid_report_json",
                id="string-version",
            ),
            pytest.param('{"summary": "a"', "invalid_report_json", id="truncated"),
        ],
    )
    def test_unusable_text_is_rejected(self, text: str, code: str) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            decode_report(text)
        assert caught.value.code == code

    def test_oversized_response_is_rejected(self) -> None:
        text = "x" * (MAX_REPORT_BYTES + 1)
        with pytest.raises(AtlasReviewError) as caught:
            decode_report(text)
        assert caught.value.code == "oversized_report"

    def test_duplicate_finding_ids_fail_the_contract(self) -> None:
        with pytest.raises(ValidationError):
            valid_report(FINDING, second_finding() | {"finding_id": "F001"})


class TestAcceptedReports:
    """A report grounded in the packet passes unchanged."""

    def test_grounded_report_is_accepted(self) -> None:
        validate_report(valid_report(), default_packet(), max_findings=5)

    def test_empty_findings_are_accepted(self) -> None:
        validate_report(valid_report(findings=[]), default_packet(), max_findings=5)

    def test_control_without_mitigation_is_accepted(self) -> None:
        control = CONTROL | {"mitigation_ids": [], "evidence_ids": ["E001"]}
        validate_report(
            valid_report(finding(controls=[control])), default_packet(), max_findings=5
        )


class TestRejectedReports:
    """Every fabricated identifier or missing link is rejected."""

    @pytest.mark.parametrize(
        ("changes", "expected"),
        [
            pytest.param(
                {"evidence_ids": ["E999"]}, "unknown evidence id E999", id="evidence"
            ),
            pytest.param(
                {"technique_ids": ["AML.T9999"]}, "unknown technique", id="technique"
            ),
            pytest.param(
                {"case_study_ids": ["AML.CS0024"]},
                "unknown case study AML.CS0024",
                id="case",
            ),
            pytest.param(
                {
                    "evidence_ids": ["E002"],
                    "case_study_ids": [],
                    "controls": [],
                },
                "no cited evidence describes",
                id="only-a-card",
            ),
            pytest.param(
                {
                    "technique_ids": [PARENT],
                    "evidence_ids": ["E007", "E006"],
                    "controls": [],
                },
                "is not linked to a cited technique",
                id="case-not-linked",
            ),
            pytest.param(
                {
                    "technique_ids": [PARENT],
                    "evidence_ids": ["E007"],
                    "case_study_ids": [],
                    "controls": [CONTROL],
                },
                "is not linked to a cited technique",
                id="mitigation-not-linked",
            ),
            pytest.param(
                {
                    "case_study_ids": [],
                    "controls": [CONTROL | {"mitigation_ids": ["AML.M9999"]}],
                },
                "unknown mitigation AML.M9999",
                id="unknown-mitigation",
            ),
            pytest.param(
                {
                    "case_study_ids": [],
                    "controls": [
                        CONTROL | {"evidence_ids": ["E999"], "mitigation_ids": []}
                    ],
                },
                "unknown evidence id E999",
                id="control-evidence",
            ),
        ],
    )
    def test_unsupported_reference_is_reported(
        self, changes: dict[str, Any], expected: str
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(
                valid_report(finding(**changes)), default_packet(), max_findings=5
            )
        assert caught.value.code == "unsupported_report_citation"
        assert expected in " ".join(caught.value.details)

    def test_case_without_an_included_description_is_rejected(self) -> None:
        without_case = packet(
            [item for item in default_items() if item.evidence_id != "E006"],
            default_associations(),
        )
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(valid_report(), without_case, max_findings=5)
        assert "has no case description" in " ".join(caught.value.details)

    def test_mitigation_without_a_description_is_rejected(self) -> None:
        cards_only = packet(
            [item for item in default_items() if item.evidence_id != "E003"],
            default_associations(),
        )
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(valid_report(), cards_only, max_findings=5)
        assert "has no description" in " ".join(caught.value.details)

    @pytest.mark.parametrize(("count", "limit"), [(2, 1), (3, 2), (5, 4)])
    def test_finding_count_limit_is_enforced(self, count: int, limit: int) -> None:
        findings = [
            FINDING
            if index == 1
            else second_finding() | {"finding_id": f"F{index:03d}"}
            for index in range(1, count + 1)
        ]
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(
                valid_report(*findings), default_packet(), max_findings=limit
            )
        assert caught.value.code == "report_finding_limit"

    def test_exercise_classification_is_not_rewritten_by_validation(self) -> None:
        validate_report(
            valid_report(finding(case_study_ids=[CASE])),
            default_packet(),
            max_findings=5,
        )
        types = {
            item.case_type for item in default_packet().items if CASE in item.entity_ids
        }
        assert types == {EXERCISE}


class TestCitedDescriptiveEvidence:
    """A named entity needs a descriptive passage inside the citations that use it."""

    def test_every_named_technique_needs_its_own_cited_description(self) -> None:
        # E001 describes TECHNIQUE only; the PARENT description E007 is in the packet.
        report = valid_report(
            finding(
                technique_ids=[TECHNIQUE, PARENT],
                evidence_ids=["E001"],
                case_study_ids=[],
                controls=[],
            )
        )
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(report, default_packet(), max_findings=5)
        assert (
            f"F001: no cited evidence describes technique {PARENT}"
            in caught.value.details
        )

    def test_case_described_only_outside_the_finding_citations_is_rejected(
        self,
    ) -> None:
        # E002 is the employs card; the E006 case narrative is in the packet but uncited.
        report = valid_report(finding(evidence_ids=["E001", "E002"], controls=[]))
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(report, default_packet(), max_findings=5)
        assert (
            f"F001: case study {CASE} has no case description among the cited evidence"
            in caught.value.details
        )

    def test_mitigation_described_only_outside_the_control_citations_is_rejected(
        self,
    ) -> None:
        # E004 is the mitigates card; the E003 mitigation text is in the packet but uncited.
        report = valid_report(
            finding(
                case_study_ids=[],
                controls=[CONTROL | {"evidence_ids": ["E004"]}],
            )
        )
        with pytest.raises(AtlasReviewError) as caught:
            validate_report(report, default_packet(), max_findings=5)
        assert (
            f"F001 control 1: mitigation {MITIGATION} has no description among the "
            "control's cited evidence" in caught.value.details
        )
