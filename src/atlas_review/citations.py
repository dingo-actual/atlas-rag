"""Trusted citation labels and canonical ATLAS routes for stored evidence.

Labels and routes are derived from stored packet records, never from model
output, so a rendered citation cannot claim more than the evidence supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from atlas_review.contracts import AllowedAssociation, EvidenceItem, EvidencePacket

_ROUTES: Mapping[str, str] = {
    "technique": "https://atlas.mitre.org/techniques/{entity_id}",
    "mitigation": "https://atlas.mitre.org/mitigations/{entity_id}",
    "case_study": "https://atlas.mitre.org/studies/{entity_id}",
}
_SUGGESTED = "none named (reviewer-suggested check, not an ATLAS-prescribed control)"


def atlas_routes(packet: EvidencePacket) -> dict[str, str]:
    """Map every allowed entity identifier to its canonical ATLAS page.

    Args:
        packet (EvidencePacket): Frozen evidence packet.

    Returns:
        (dict[str, str]) Routes keyed by ATLAS identifier; unknown kinds are omitted.
    """
    return {
        entity_id: template.format(entity_id=entity_id)
        for kind, entities in packet.allowed_entity_ids_by_kind.items()
        for entity_id in entities
        if (template := _ROUTES.get(kind)) is not None
    }


def case_label(case_id: str, items: Sequence[EvidenceItem]) -> str:
    """Return ``ID (classification)`` from the stored case type, if any.

    Args:
        case_id (str): ATLAS case-study identifier.
        items (Sequence[EvidenceItem]): Evidence items that may carry its type.

    Returns:
        (str) Label with the stored classification, or the bare identifier.
    """
    case_type = next(
        (
            item.case_type
            for item in items
            if case_id in item.entity_ids and item.case_type is not None
        ),
        None,
    )
    return case_id if case_type is None else f"{case_id} ({case_type})"


def mitigation_labels(
    mitigation_ids: Sequence[str],
    technique_ids: Sequence[str],
    associations: Sequence[AllowedAssociation],
) -> str:
    """Label named mitigations as direct, parent-linked, or reviewer-suggested.

    Args:
        mitigation_ids (Sequence[str]): Mitigations the control names.
        technique_ids (Sequence[str]): Techniques of the owning finding.
        associations (Sequence[AllowedAssociation]): Packet association records.

    Returns:
        (str) Comma-separated labels for every named mitigation.
    """
    if not mitigation_ids:
        return _SUGGESTED
    return ", ".join(
        _mitigation_label(mitigation_id, technique_ids, associations)
        for mitigation_id in mitigation_ids
    )


def _mitigation_label(
    mitigation_id: str,
    technique_ids: Sequence[str],
    associations: Sequence[AllowedAssociation],
) -> str:
    """Return one mitigation with its recorded association kind."""
    for association in associations:
        if (
            association.source_kind == "mitigation"
            and association.source_id == mitigation_id
            and association.technique_id in technique_ids
        ):
            parent = association.parent_technique_id
            return (
                f"{mitigation_id} (direct)"
                if parent is None
                else f"{mitigation_id} (parent-linked via {parent})"
            )
    return mitigation_id
