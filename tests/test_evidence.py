"""Evidence closure: complete units, connector cards, and allowlists."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from atlas_review.config import AppConfig, RetrievalConfig
from atlas_review.contracts import (
    Candidate,
    EvidencePacket,
    Passage,
    RetrievalTrace,
)
from atlas_review.retrieval.embeddings import EmbeddingBackend
from atlas_review.retrieval.evidence import select_evidence
from atlas_review.retrieval.loader import LoadedGeneration
from atlas_review.retrieval.pipeline import RetrievalRun, retrieve
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.storage.codec import decode_model, encode_model
from fakes import RecordingReranker, WhitespaceTokenCounter

FOCUS = "indirect prompt injection through retrieved content"
CASE = "AML.CS0035"
TECHNIQUE = "AML.T0051.001"
PARENT = "AML.T0051"
MITIGATION = "AML.M0033"
COUNTERS = TokenCounters(
    embedding=WhitespaceTokenCounter(), reranker=WhitespaceTokenCounter()
)


def _retrieve(
    loaded_generation: LoadedGeneration,
    embedder: EmbeddingBackend,
    reranker: RecordingReranker,
    config: AppConfig | None = None,
    focus: str = FOCUS,
) -> RetrievalRun:
    """Run the retrieval pipeline over the built fixture."""
    return retrieve(
        loaded_generation, embedder, reranker, config or AppConfig(), focus, COUNTERS
    )


def _config(**change: int) -> AppConfig:
    """Return the default configuration with explicit retrieval overrides."""
    return AppConfig.model_validate(
        {
            **AppConfig().model_dump(mode="python"),
            "retrieval": {
                **RetrievalConfig().model_dump(mode="python"),
                **change,
            },
        },
        strict=False,
    )


@pytest.fixture
def run(
    loaded_generation: LoadedGeneration,
    embedder: EmbeddingBackend,
    reranker: RecordingReranker,
) -> RetrievalRun:
    """Return one default retrieval run over the fixture generation."""
    return _retrieve(loaded_generation, embedder, reranker)


def _passages_by_kind(loaded_generation: LoadedGeneration, kind: str) -> list[Passage]:
    """Return the stored passages of one owner kind in stable order."""
    return [
        passage
        for passage in loaded_generation.passages.values()
        if passage.entity_id is not None
        and loaded_generation.graph.entity(passage.entity_id).kind == kind
    ]


def _rank(passages: Sequence[Passage]) -> list[Candidate]:
    """Build a dense-only candidate list in the given passage order."""
    return [
        Candidate(
            passage_id=passage.passage_id,
            channels=["dense"],
            dense_rank=index,
            dense_distance=0.1,
            fusion_score=0.01,
        )
        for index, passage in enumerate(passages, start=1)
    ]


class TestEvidenceBudgets:
    """Whole units are accepted only inside the documented budgets."""

    def test_evidence_ids_are_sequential_labels(self, run: RetrievalRun) -> None:
        identifiers = [item.evidence_id for item in run.selection.packet.items]
        assert identifiers == [
            f"E{index:03d}" for index in range(1, len(identifiers) + 1)
        ]

    def test_item_and_character_limits_hold(self, run: RetrievalRun) -> None:
        config = AppConfig()
        assert len(run.selection.packet.items) <= config.retrieval.evidence_item_limit
        characters = sum(len(item.text) for item in run.selection.packet.items)
        assert characters <= config.retrieval.evidence_character_limit

    def test_each_owner_contributes_at_most_one_passage_per_limit(
        self, loaded_generation: LoadedGeneration
    ) -> None:
        passages = list(loaded_generation.passages.values())
        selection = select_evidence(
            _rank(passages),
            graph=loaded_generation.graph,
            passages=loaded_generation.passages,
            generation_id=loaded_generation.generation_id,
            collection_version=loaded_generation.collection_version,
            config=_config(per_owner_passage_limit=1),
        )
        selected = [
            item.passage_id
            for item in selection.packet.items
            if item.passage_id is not None
        ]
        owners = [
            loaded_generation.passages[passage_id].entity_id
            or loaded_generation.passages[passage_id].edge_id
            for passage_id in selected
        ]
        assert len(owners) == len(set(owners))

    def test_character_budget_rejects_whole_units(
        self, loaded_generation: LoadedGeneration
    ) -> None:
        selection = select_evidence(
            _rank(list(loaded_generation.passages.values())),
            graph=loaded_generation.graph,
            passages=loaded_generation.passages,
            generation_id=loaded_generation.generation_id,
            collection_version=loaded_generation.collection_version,
            config=_config(evidence_character_limit=250),
        )
        characters = sum(len(item.text) for item in selection.packet.items)
        assert characters <= 250
        assert selection.exclusions

    def test_missing_categories_are_flagged(
        self, loaded_generation: LoadedGeneration
    ) -> None:
        techniques = _passages_by_kind(loaded_generation, "technique")
        selection = select_evidence(
            _rank(techniques),
            graph=loaded_generation.graph,
            passages=loaded_generation.passages,
            generation_id=loaded_generation.generation_id,
            collection_version=loaded_generation.collection_version,
            config=AppConfig(),
        )
        notes = " ".join(selection.missing_categories)
        assert "case-study" in notes
        assert "mitigation" in notes


class TestEvidenceSourceBoundary:
    """Generated context ranks passages but never enters the packet."""

    def test_passage_text_is_the_original_excerpt(
        self, run: RetrievalRun, loaded_generation: LoadedGeneration
    ) -> None:
        for item in run.selection.packet.items:
            if item.passage_id is None:
                continue
            passage = loaded_generation.passages[item.passage_id]
            assert item.text.endswith(passage.source_text)
            assert passage.context_text not in item.text
            assert passage.retrieval_text not in item.text

    def test_relationship_passage_carries_a_structural_card(
        self, run: RetrievalRun, loaded_generation: LoadedGeneration
    ) -> None:
        packet = run.selection.packet
        cards = {item.edge_id for item in packet.items if item.passage_id is None}
        for item in packet.items:
            if item.passage_id is None:
                continue
            passage = loaded_generation.passages[item.passage_id]
            if passage.edge_id is not None:
                assert passage.edge_id in cards

    def test_graph_derived_item_keeps_every_connector(self, run: RetrievalRun) -> None:
        cards = {
            item.edge_id
            for item in run.selection.packet.items
            if item.passage_id is None
        }
        for item in run.selection.packet.items:
            if item.graph_path is None:
                continue
            for hop in item.graph_path.hops:
                assert hop.edge_id in cards

    def test_connector_cards_are_structural_only(self, run: RetrievalRun) -> None:
        for item in run.selection.packet.items:
            if item.passage_id is not None:
                continue
            assert item.item_kind == "relationship"
            assert item.edge_id is not None
            assert item.retrieval_channels == ["graph_connector"]


class TestAllowedAssociations:
    """Direct and parent-linked source relationships stay distinguishable."""

    def test_direct_mitigation_association_is_recorded(self, run: RetrievalRun) -> None:
        associations = run.selection.packet.allowed_associations
        assert any(
            association.source_id == MITIGATION
            and association.technique_id == PARENT
            and association.parent_technique_id is None
            for association in associations
        )

    def test_parent_link_is_labeled_as_a_parent(self, run: RetrievalRun) -> None:
        associations = run.selection.packet.allowed_associations
        assert any(
            association.source_id == MITIGATION
            and association.technique_id == TECHNIQUE
            and association.parent_technique_id == PARENT
            for association in associations
        )
        assert not any(
            association.source_id == MITIGATION
            and association.technique_id == TECHNIQUE
            and association.parent_technique_id is None
            for association in associations
        )

    def test_associations_cite_evidence_inside_the_packet(
        self, run: RetrievalRun
    ) -> None:
        identifiers = {item.evidence_id for item in run.selection.packet.items}
        for association in run.selection.packet.allowed_associations:
            assert set(association.evidence_ids) <= identifiers
            assert association.edge_ids

    def test_entity_allowlist_only_names_packet_entities(
        self, run: RetrievalRun
    ) -> None:
        packet = run.selection.packet
        named = {entity for item in packet.items for entity in item.entity_ids}
        for kind, identifiers in packet.allowed_entity_ids_by_kind.items():
            assert kind in {"technique", "mitigation", "case_study"}
            assert set(identifiers) <= named


class TestSelectionDeterminism:
    """Reranked order drives selection and is preserved in the trace."""

    def test_selected_items_follow_the_reranked_order(self, run: RetrievalRun) -> None:
        selected_ids = {
            candidate.passage_id for candidate in run.ranked if candidate.selected
        }
        packet_ids = {
            item.passage_id
            for item in run.selection.packet.items
            if item.passage_id is not None
        }
        assert packet_ids == selected_ids

    def test_selection_is_repeatable(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        config = _config(dense_k=2, bm25_k=2)
        first = _retrieve(loaded_generation, embedder, reranker, config)
        second = _retrieve(loaded_generation, embedder, reranker, config)
        assert first.selection.packet == second.selection.packet
        assert first.selection.provenance == second.selection.provenance


class TestPacketPersistence:
    """The frozen packet survives the storage adapter unchanged."""

    def test_packet_round_trips_through_the_msgspec_adapter(
        self, run: RetrievalRun
    ) -> None:
        packet = run.selection.packet
        restored = decode_model(encode_model(packet), EvidencePacket)
        assert restored == packet
        assert restored.items

    def test_trace_round_trips_through_the_msgspec_adapter(
        self, run: RetrievalRun
    ) -> None:
        restored = decode_model(encode_model(run.trace), RetrievalTrace)
        assert restored == run.trace

    def test_restored_packet_excludes_generated_context(
        self, run: RetrievalRun, loaded_generation: LoadedGeneration
    ) -> None:
        restored = decode_model(encode_model(run.selection.packet), EvidencePacket)
        contexts = {
            passage.context_text for passage in loaded_generation.passages.values()
        }
        assert all(item.text not in contexts for item in restored.items)
        assert all(
            item.edge_id is not None or item.passage_id is not None
            for item in restored.items
        )
