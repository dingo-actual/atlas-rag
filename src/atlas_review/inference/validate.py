"""Strict decoding and evidence validation of synthesized review reports."""

from __future__ import annotations

from pydantic import ValidationError

from atlas_review.contracts import (
    ControlRecommendation,
    EvidenceItem,
    EvidencePacket,
    ReviewFinding,
    ReviewReport,
)
from atlas_review.errors import (
    STAGE_LLM_OUTPUT,
    AtlasReviewError,
    validation_details,
)

MAX_REPORT_BYTES = 100_000
_MAX_DETAILS = 20


def decode_report(text: str) -> ReviewReport:
    """Validate provider text as a bounded, strict review report.

    Args:
        text (str): Raw response text from the configured provider.

    Returns:
        (ReviewReport) Validated report.

    Raises:
        AtlasReviewError: The text is oversized or fails strict validation.
    """
    if len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise AtlasReviewError(
            code="oversized_report",
            stage=STAGE_LLM_OUTPUT,
            message=f"the provider response exceeds {MAX_REPORT_BYTES} bytes",
            recovery="Reduce the configured output allowance, then retry explicitly.",
        )
    try:
        return ReviewReport.model_validate_json(text, strict=True)
    except ValidationError as exc:
        raise AtlasReviewError(
            code="invalid_report_json",
            stage=STAGE_LLM_OUTPUT,
            message="the provider response is not a valid review report",
            details=validation_details(exc),
            recovery="Retry the generation or adjust the provider settings.",
        ) from exc


def validate_report(
    report: ReviewReport, packet: EvidencePacket, *, max_findings: int
) -> None:
    """Check every identifier and relationship the report claims.

    Args:
        report (ReviewReport): Validated synthesis output.
        packet (EvidencePacket): Frozen evidence packet supplied to the model.
        max_findings (int): Configured largest finding count.

    Raises:
        AtlasReviewError: A finding exceeds the limit, or cites an identifier
            or association that this packet does not support.
    """
    if len(report.findings) > max_findings:
        raise AtlasReviewError(
            code="report_finding_limit",
            stage=STAGE_LLM_OUTPUT,
            message=(
                f"the report has {len(report.findings)} findings; the limit is "
                f"{max_findings}"
            ),
            recovery="Retry the generation, or raise the configured finding limit.",
        )
    items = {item.evidence_id: item for item in packet.items}
    errors = [
        error
        for finding in report.findings
        for error in _finding_errors(finding, packet, items)
    ]
    if errors:
        raise AtlasReviewError(
            code="unsupported_report_citation",
            stage=STAGE_LLM_OUTPUT,
            message="the report cites evidence or relationships this packet cannot support",
            details=tuple(errors[:_MAX_DETAILS]),
            recovery="Retry the generation, or edit the brief and retrieve again.",
        )


def _finding_errors(
    finding: ReviewFinding, packet: EvidencePacket, items: dict[str, EvidenceItem]
) -> list[str]:
    """Return every citation and relationship problem in one finding."""
    errors = _citation_errors(finding, packet, items)
    errors.extend(_case_errors(finding, packet, items))
    for position, control in enumerate(finding.controls, start=1):
        errors.extend(_control_errors(finding, control, str(position), packet, items))
    return errors


def _citation_errors(
    finding: ReviewFinding, packet: EvidencePacket, items: dict[str, EvidenceItem]
) -> list[str]:
    """Return unknown identifiers and missing technique support."""
    techniques = set(packet.allowed_entity_ids_by_kind.get("technique", ()))
    errors = [
        f"{finding.finding_id}: unknown evidence id {evidence_id}"
        for evidence_id in finding.evidence_ids
        if evidence_id not in items
    ]
    errors.extend(
        f"{finding.finding_id}: unknown technique {technique_id}"
        for technique_id in finding.technique_ids
        if technique_id not in techniques
    )
    cited = [
        items[evidence_id]
        for evidence_id in finding.evidence_ids
        if evidence_id in items
    ]
    if not any(
        _describes(item, technique_id)
        for item in cited
        for technique_id in finding.technique_ids
    ):
        errors.append(
            f"{finding.finding_id}: no cited evidence describes a listed technique"
        )
    return errors


def _case_errors(
    finding: ReviewFinding, packet: EvidencePacket, items: dict[str, EvidenceItem]
) -> list[str]:
    """Return unknown or unsupported case-study claims."""
    allowed = set(packet.allowed_entity_ids_by_kind.get("case_study", ()))
    errors: list[str] = []
    for case_id in finding.case_study_ids:
        if case_id not in allowed:
            errors.append(f"{finding.finding_id}: unknown case study {case_id}")
        elif not any(_describes(item, case_id) for item in items.values()):
            errors.append(
                f"{finding.finding_id}: case study {case_id} has no case description "
                "in this packet"
            )
        elif not _linked(packet, "case_study", case_id, finding.technique_ids):
            errors.append(
                f"{finding.finding_id}: case study {case_id} is not linked to a "
                "cited technique in this packet"
            )
    return errors


def _control_errors(
    finding: ReviewFinding,
    control: ControlRecommendation,
    position: str,
    packet: EvidencePacket,
    items: dict[str, EvidenceItem],
) -> list[str]:
    """Return unknown or unsupported control claims for one control."""
    label = f"{finding.finding_id} control {position}"
    errors = [
        f"{label}: unknown evidence id {evidence_id}"
        for evidence_id in control.evidence_ids
        if evidence_id not in items
    ]
    allowed = set(packet.allowed_entity_ids_by_kind.get("mitigation", ()))
    for mitigation_id in control.mitigation_ids:
        errors.extend(
            _mitigation_errors(label, mitigation_id, allowed, finding, packet, items)
        )
    return errors


def _mitigation_errors(
    label: str,
    mitigation_id: str,
    allowed: set[str],
    finding: ReviewFinding,
    packet: EvidencePacket,
    items: dict[str, EvidenceItem],
) -> list[str]:
    """Return the problem with one named mitigation, if any."""
    if mitigation_id not in allowed:
        return [f"{label}: unknown mitigation {mitigation_id}"]
    if not _linked(packet, "mitigation", mitigation_id, finding.technique_ids):
        return [
            (
                f"{label}: mitigation {mitigation_id} is not linked to a "
                "cited technique in this packet"
            )
        ]
    if not any(_describes(item, mitigation_id) for item in items.values()):
        return [
            f"{label}: mitigation {mitigation_id} has no description in this packet"
        ]
    return []


def _describes(item: EvidenceItem, entity_id: str) -> bool:
    """Return whether a passage item carries substantive text for an entity."""
    return item.item_kind == "passage" and entity_id in item.entity_ids


def _linked(
    packet: EvidencePacket,
    source_kind: str,
    source_id: str,
    technique_ids: list[str],
) -> bool:
    """Return whether an allowed association connects the source to a technique."""
    return any(
        association.source_kind == source_kind
        and association.source_id == source_id
        and association.technique_id in technique_ids
        for association in packet.allowed_associations
    )
