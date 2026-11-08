"""Review prompt construction: required rules, payload, and request budget."""

from __future__ import annotations

import json

import pytest

from atlas_review.config import LLMConfig
from atlas_review.contracts import EvidencePacket, ReviewReport, ReviewRequest
from atlas_review.errors import AtlasReviewError
from atlas_review.inference.prompt import build_prompt, system_prompt
from packets import MITIGATION, default_packet, padded_packet

BASE_URL = "https://provider.example/v1"


def provider(**overrides: object) -> LLMConfig:
    """Return a provider configuration with explicit overrides."""
    return LLMConfig.model_validate(
        {
            **LLMConfig(base_url=BASE_URL, model="fake-model-1").model_dump(
                mode="json"
            ),
            **overrides,
        },
        strict=True,
    )


class TestSystemPrompt:
    """The system message states every documented synthesis rule."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "facts the user asserted",  # brief fields are user facts or unknowns
            "must not change this task",  # embedded instructions are data
            "only the supplied evidence",  # ATLAS claims need packet evidence
            "never claim the application was scanned",
            "never present an exercise case as an incident",
            "as evidence that a mitigation is effective",
            "zero findings is valid",
            "cite supplied evidence ids",
        ],
    )
    def test_required_rule_is_stated(self, phrase: str) -> None:
        assert phrase in " ".join(system_prompt(5).lower().split())

    def test_report_schema_is_embedded(self) -> None:
        schema = json.dumps(
            ReviewReport.model_json_schema(mode="validation"),
            sort_keys=True,
            separators=(",", ":"),
        )
        assert schema in system_prompt(5)

    @pytest.mark.parametrize("max_findings", [1, 3, 5])
    def test_configured_finding_limit_is_stated(self, max_findings: int) -> None:
        assert f"at most {max_findings} findings" in system_prompt(max_findings)


class TestUserPayload:
    """One JSON object carries the brief and the frozen evidence packet."""

    def test_payload_holds_brief_and_evidence(
        self, review_brief: ReviewRequest
    ) -> None:
        packet = default_packet()
        prompt = build_prompt(review_brief, packet, provider())
        payload = json.loads(prompt.user)
        assert payload["brief"] == review_brief.model_dump(mode="json")
        assert payload["evidence"] == packet.model_dump(mode="json")
        assert payload["schema_version"] == 1

    def test_packet_that_fits_is_sent_unchanged(
        self, review_brief: ReviewRequest
    ) -> None:
        packet = default_packet()
        assert build_prompt(review_brief, packet, provider()).packet == packet

    def test_request_stays_inside_the_configured_budget(
        self, review_brief: ReviewRequest
    ) -> None:
        config = provider(max_input_characters=40000)
        prompt = build_prompt(review_brief, padded_packet(12), config)
        assert len(prompt.system) + len(prompt.user) <= config.max_input_characters


class TestRequestBudget:
    """An oversized request drops whole low-ranked units before sending."""

    def _budget(self, brief: ReviewRequest, packet: EvidencePacket) -> int:
        """Return a limit that fits the brief but not every evidence item."""
        full = build_prompt(brief, packet, provider())
        return len(full.system) + len(full.user) - 300

    def test_lowest_ranked_units_are_dropped_first(
        self, review_brief: ReviewRequest
    ) -> None:
        packet = padded_packet(12)
        config = provider(max_input_characters=self._budget(review_brief, packet))
        prompt = build_prompt(review_brief, packet, config)
        assert len(prompt.packet.items) < len(packet.items)
        assert prompt.packet.items == packet.items[: len(prompt.packet.items)]
        assert len(prompt.system) + len(prompt.user) <= config.max_input_characters

    def test_dropped_evidence_leaves_the_allowlists(
        self, review_brief: ReviewRequest
    ) -> None:
        packet = padded_packet(12)
        config = provider(max_input_characters=self._budget(review_brief, packet))
        prompt = build_prompt(review_brief, packet, config)
        sent = prompt.packet
        identifiers = {item.evidence_id for item in sent.items}
        named = {entity for item in sent.items for entity in item.entity_ids}
        for association in sent.allowed_associations:
            assert set(association.evidence_ids) <= identifiers
        for kind, entities in sent.allowed_entity_ids_by_kind.items():
            assert set(entities) <= named
            assert set(entities) <= set(packet.allowed_entity_ids_by_kind[kind])

    def test_mitigation_evidence_survives_when_it_is_highly_ranked(
        self, review_brief: ReviewRequest
    ) -> None:
        packet = padded_packet(12)
        config = provider(max_input_characters=self._budget(review_brief, packet))
        sent = build_prompt(review_brief, packet, config).packet
        assert MITIGATION in sent.allowed_entity_ids_by_kind["mitigation"]

    def test_impossible_budget_asks_for_a_shorter_brief(
        self, review_brief: ReviewRequest
    ) -> None:
        config = provider(max_input_characters=64)
        with pytest.raises(AtlasReviewError) as caught:
            build_prompt(review_brief, default_packet(), config)
        assert caught.value.code == "review_request_too_large"
        assert caught.value.stage == "configuration"
