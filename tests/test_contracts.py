"""Contract tests: strict Pydantic policy and the canonical source records."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from atlas_review.contracts import (
    AtlasEntity,
    AtlasRelationship,
    ReviewRequest,
    SourceSnapshot,
)

EXAMPLE_BRIEF: dict[str, Any] = {
    "schema_version": 1,
    "title": "Internal knowledge assistant design review",
    "review_question": "How could untrusted retrieved content influence sensitive answers?",
    "retrieval_focus": "Indirect prompt injection through retrieved workspace messages",
    "system_description": "An employee assistant retrieves documents and returns a cited answer.",
    "users_and_roles": ["Employees", "External collaborators"],
    "data_sources": ["Internal document repository"],
    "untrusted_inputs": ["Messages written by external collaborators"],
    "tools_and_permissions": ["Search documents"],
    "sensitive_assets": ["Internal documents"],
    "existing_controls": ["Employees authenticate before use"],
    "constraints": ["No action tools beyond retrieval"],
    "unknowns": ["Whether document authorization is checked per employee"],
}

VALID_SNAPSHOT: dict[str, Any] = {
    "origin": "local:fixtures/mini_atlas_v6.yaml",
    "format_version": "6.0.0",
    "collection_version": "2026.09",
    "acquired_at": "2026-10-26T00:00:00Z",
    "source_sha256": "a" * 64,
    "commit": "3259f388d19cbcca11bacf12a0ef97f4198f711b",
    "license": "Apache-2.0",
}

VALID_ENTITY: dict[str, Any] = {
    "generation_id": "generation-1",
    "entity_id": "AML.CS0035",
    "kind": "case_study",
    "name": "Data Exfiltration from Slack AI",
    "display_name": "Data Exfiltration from Slack AI",
    "description": "An exercise case.",
    "source_locator": "/case-studies/AML.CS0035/description",
    "references": [
        {"ref_id": "ref-1", "title": "Reference", "url": "https://example.invalid"}
    ],
    "created_date": "2024-08-20",
    "case_type": "Exercise",
    "case_actor": "PromptArmor",
    "case_date_granularity": "Day",
}

VALID_EDGE: dict[str, Any] = {
    "generation_id": "generation-1",
    "edge_id": "c" * 64,
    "relation": "employs",
    "source_id": "AML.CS0035",
    "target_id": "AML.T0012",
    "description": "The researcher created a non-admin account.",
    "step_id": "S02",
    "tactic_id": "AML.TA0004",
    "leads_to": ["S03"],
    "source_locator": "/relationships/AML.CS0035/employs/1/description",
}


def brief_with(**overrides: Any) -> dict[str, Any]:
    return {**EXAMPLE_BRIEF, **overrides}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"schema_version": True}, "schema_version", id="boolean-version"),
        pytest.param({"schema_version": 1.0}, "schema_version", id="float-version"),
        pytest.param({"schema_version": "1"}, "schema_version", id="string-version"),
        pytest.param({"unexpected": "value"}, "extra_forbidden", id="extra-field"),
        pytest.param({"title": ""}, "title", id="empty-title"),
        pytest.param({"unknowns": ["z"] * 21}, "20", id="too-many-entries"),
        pytest.param(
            {"system_description": "x" * 11_950, "review_question": "y" * 100},
            "12000",
            id="over-text-budget",
        ),
    ],
)
def test_review_request_rejects_invalid_payloads(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        ReviewRequest.model_validate(brief_with(**overrides), strict=True)


def test_review_request_drops_missing_required_field() -> None:
    incomplete = {
        key: value for key, value in EXAMPLE_BRIEF.items() if key != "retrieval_focus"
    }
    with pytest.raises(ValidationError, match="retrieval_focus"):
        ReviewRequest.model_validate(incomplete, strict=True)


def test_review_request_round_trips_the_specification_example() -> None:
    request = ReviewRequest.model_validate_json(json.dumps(EXAMPLE_BRIEF), strict=True)
    assert request.schema_version == 1
    assert request.users_and_roles == ["Employees", "External collaborators"]
    assert json.loads(request.model_dump_json())["schema_version"] == 1


def test_contracts_are_frozen() -> None:
    request = ReviewRequest.model_validate(EXAMPLE_BRIEF, strict=True)
    with pytest.raises(ValidationError):
        request.title = "changed"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"source_sha256": "not-a-digest"}, "SHA-256", id="bad-digest"),
        pytest.param({"origin": "  "}, "origin", id="blank-origin"),
        pytest.param({"acquired_at": "yesterday"}, "acquired_at", id="bad-timestamp"),
        pytest.param({"format_version": ""}, "format_version", id="blank-format"),
    ],
)
def test_source_snapshot_rejects_invalid_identity(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        SourceSnapshot.model_validate({**VALID_SNAPSHOT, **overrides}, strict=True)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"kind": "widget"}, "kind", id="unknown-kind"),
        pytest.param({"entity_id": ""}, "entity_id", id="empty-entity-id"),
        pytest.param({"description": ""}, "description", id="empty-description"),
        pytest.param(
            {"source_locator": "not-a-path"}, "source_locator", id="bad-locator"
        ),
        pytest.param({"accepts_unexpected": True}, "extra_forbidden", id="extra-field"),
    ],
)
def test_atlas_entity_rejects_invalid_records(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        AtlasEntity.model_validate({**VALID_ENTITY, **overrides}, strict=True)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"relation": "supports"}, "relation", id="unknown-relation"),
        pytest.param({"edge_id": "short"}, "edge_id", id="bad-edge-id"),
        pytest.param({"target_id": ""}, "target_id", id="empty-target"),
        pytest.param({"step_id": "  "}, "step_id", id="blank-step"),
    ],
)
def test_atlas_relationship_rejects_invalid_records(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        AtlasRelationship.model_validate({**VALID_EDGE, **overrides}, strict=True)


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            SourceSnapshot.model_validate(VALID_SNAPSHOT, strict=True), id="snapshot"
        ),
        pytest.param(
            AtlasEntity.model_validate(VALID_ENTITY, strict=True), id="entity"
        ),
        pytest.param(
            AtlasRelationship.model_validate(VALID_EDGE, strict=True), id="edge"
        ),
    ],
)
def test_canonical_records_accept_documented_values(record: Any) -> None:
    assert record.model_dump(mode="json")[
        "generation_id" if hasattr(record, "generation_id") else "origin"
    ]
