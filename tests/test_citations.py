"""Trusted citation labels and canonical ATLAS routes for stored evidence."""

from __future__ import annotations

import pytest

from atlas_review.citations import atlas_routes, case_label, mitigation_labels
from atlas_review.contracts import AllowedAssociation, EvidencePacket
from packets import (
    CASE,
    EXERCISE,
    MITIGATION,
    PARENT,
    TECHNIQUE,
    default_associations,
    default_packet,
    digest,
)


class TestAtlasRoutes:
    """Every allowed entity resolves to its canonical ATLAS page."""

    @pytest.mark.parametrize(
        ("entity_id", "route"),
        [
            pytest.param(
                TECHNIQUE,
                f"https://atlas.mitre.org/techniques/{TECHNIQUE}",
                id="technique",
            ),
            pytest.param(
                PARENT,
                f"https://atlas.mitre.org/techniques/{PARENT}",
                id="parent-technique",
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
    def test_allowed_entity_gets_its_canonical_page(
        self, entity_id: str, route: str
    ) -> None:
        assert atlas_routes(default_packet())[entity_id] == route

    def test_packet_without_entities_has_no_routes(self) -> None:
        empty = EvidencePacket(
            schema_version=1,
            generation_id="generation-1",
            collection_version="2026.09",
        )
        assert atlas_routes(empty) == {}


class TestCaseLabel:
    """Case studies keep the classification stored in the packet."""

    def test_stored_classification_is_appended(self) -> None:
        packet = default_packet()
        assert case_label(CASE, packet.items) == f"{CASE} ({EXERCISE})"

    def test_case_without_a_classification_keeps_its_identifier(self) -> None:
        packet = default_packet()
        assert case_label("AML.CS9999", packet.items) == "AML.CS9999"


class TestMitigationLabels:
    """A mitigation is labelled by the association the packet records."""

    def test_parent_linked_association_is_not_called_direct(self) -> None:
        label = mitigation_labels([MITIGATION], [TECHNIQUE], default_associations())
        assert label == f"{MITIGATION} (parent-linked via {PARENT})"

    def test_direct_association_is_labelled_direct(self) -> None:
        direct = AllowedAssociation(
            source_id=MITIGATION,
            source_kind="mitigation",
            technique_id=TECHNIQUE,
            edge_ids=[digest("mitigates")],
            evidence_ids=["E003"],
        )
        assert mitigation_labels([MITIGATION], [TECHNIQUE], [direct]) == (
            f"{MITIGATION} (direct)"
        )

    def test_association_for_another_technique_does_not_apply(self) -> None:
        assert (
            mitigation_labels([MITIGATION], ["AML.T0001"], default_associations())
            == MITIGATION
        )

    def test_no_named_mitigation_is_a_reviewer_suggested_check(self) -> None:
        assert "not an ATLAS-prescribed control" in mitigation_labels(
            [], [TECHNIQUE], default_associations()
        )
