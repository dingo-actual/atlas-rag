"""ATLAS 6.1.0 records: software, entities, and the four new relations.

New kinds and relations are stored and inspectable. They are deliberately not
chunked, embedded, seeded, or traversed, so retrieval semantics stay unchanged.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.contracts import AssociatedEntity
from atlas_review.corpus.atlas_v6 import ParsedCorpus, parse_source
from atlas_review.corpus.chunk import build_source_chunks
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.graph_index import build_graph
from atlas_review.retrieval.graph_search import (
    extract_exact_ids,
    select_seeds,
    traverse,
)
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.services.session import WorkspaceSession

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "mini_atlas_v6_1.yaml"
LEGACY_FIXTURE = ROOT / "fixtures" / "mini_atlas_v6.yaml"
ACQUIRED_AT = "2026-11-01T00:00:00+00:00"
NEW_RELATIONS = ("uses", "attributed-to", "implements", "reported-by")
SEARCHABLE_RELATIONS = {"employs", "mitigates", "specializes", "achieves", "sequences"}


@pytest.fixture(scope="module")
def corpus() -> ParsedCorpus:
    """Return the 6.1.0 fixture parsed once."""
    return parse_source(
        FIXTURE.read_bytes(), origin=str(FIXTURE), acquired_at=ACQUIRED_AT
    )


@pytest.fixture(scope="module")
def legacy_corpus() -> ParsedCorpus:
    """Return the 6.0.0 fixture parsed once."""
    return parse_source(
        LEGACY_FIXTURE.read_bytes(), origin=str(LEGACY_FIXTURE), acquired_at=ACQUIRED_AT
    )


def test_new_sections_are_parsed_with_typed_fields(corpus: ParsedCorpus) -> None:
    """Software and entity records keep their own metadata."""
    kinds = Counter(entity.kind for entity in corpus.entities)
    assert kinds["software"] == 2
    assert kinds["entity"] == 2
    software = next(item for item in corpus.entities if item.entity_id == "AML.S0001")
    assert software.kind == "software"
    assert software.software_type == "Tool"
    assert software.software_categories == ["program"]
    assert software.aliases == ["metame-cli"]
    entity = next(item for item in corpus.entities if item.entity_id == "AML.E0001")
    assert entity.kind == "entity"
    assert entity.entity_type == "Team"
    assert entity.associated_entities == [
        AssociatedEntity(entity_id="AML.E0002", name="PromptArmor")
    ]
    assert corpus.snapshot.format_version == "6.1.0"
    assert corpus.snapshot.collection_version == "2026.10"


@pytest.mark.parametrize(
    ("relation", "source_id", "target_id"),
    [
        pytest.param("uses", "AML.CS0035", "AML.S0001", id="uses-case"),
        pytest.param("uses", "AML.E0001", "AML.S0002", id="uses-entity-tool"),
        pytest.param("uses", "AML.E0002", "AML.T0051", id="uses-entity-technique"),
        pytest.param("attributed-to", "AML.CS0035", "AML.E0002", id="attributed-to"),
        pytest.param("implements", "AML.S0001", "AML.T0051.001", id="implements"),
        pytest.param("reported-by", "AML.CS0035", "AML.E0001", id="reported-by"),
    ],
)
def test_new_relations_connect_the_declared_endpoints(
    corpus: ParsedCorpus, relation: str, source_id: str, target_id: str
) -> None:
    """Each new relationship type is stored with its source orientation."""
    found = [
        edge
        for edge in corpus.edges
        if edge.relation == relation
        and edge.source_id == source_id
        and edge.target_id == target_id
    ]
    assert found
    assert found[0].source_locator.startswith("/relationships/")


def test_activity_purpose_is_a_typed_edge_field(corpus: ParsedCorpus) -> None:
    """``activity-purpose`` is kept without an additive-field warning."""
    uses = next(edge for edge in corpus.edges if edge.relation == "uses")
    assert uses.activity_purpose == "Security Evaluation"
    assert uses.source_fields == {}
    assert not any("activity-purpose" in warning for warning in corpus.warnings)


def test_a_6_0_document_without_the_new_sections_still_parses(
    legacy_corpus: ParsedCorpus,
) -> None:
    """The 6.1.0 sections stay optional for older documents."""
    assert not any(
        entity.kind in {"software", "entity"} for entity in legacy_corpus.entities
    )
    assert not any(edge.relation in NEW_RELATIONS for edge in legacy_corpus.edges)


def test_an_unknown_relation_type_is_still_refused() -> None:
    """Adding named types does not turn off unknown-type errors."""
    raw = FIXTURE.read_bytes().replace(b"uses:", b"owns:", 1)
    with pytest.raises(AtlasReviewError) as caught:
        parse_source(raw, origin=str(FIXTURE), acquired_at=ACQUIRED_AT)
    assert caught.value.code == "unsupported_relationship_type"


def test_a_new_relation_with_the_wrong_endpoint_is_refused() -> None:
    """``uses`` refuses a pair that the source never declares."""
    raw = FIXTURE.read_bytes().replace(
        b"        target: AML.S0001\n        relationship-type: uses",
        b"        target: AML.M0033\n        relationship-type: uses",
        1,
    )
    assert raw != FIXTURE.read_bytes()
    with pytest.raises(AtlasReviewError) as caught:
        parse_source(raw, origin=str(FIXTURE), acquired_at=ACQUIRED_AT)
    assert caught.value.code == "relationship_endpoint_kind"


def test_new_kinds_and_relations_round_trip_through_storage(
    fake_session: WorkspaceSession,
) -> None:
    """Schema version 3 accepts the new kinds and relations."""
    outcome = fake_session.ingest_file(FIXTURE)
    assert outcome.generation_id
    software = fake_session.inspect("AML.S0001")
    assert software.entity.kind == "software"
    assert software.entity.software_type == "Tool"
    assert [edge.relation for edge in software.outgoing] == ["implements"]
    entity = fake_session.inspect("AML.E0001")
    assert entity.entity.kind == "entity"
    assert entity.entity.entity_type == "Team"
    assert {edge.relation for edge in entity.outgoing} == {"uses"}
    assert {edge.relation for edge in entity.incoming} == {"reported-by"}
    assert {
        edge.relation for edge in fake_session.repository.edges(outcome.generation_id)
    } >= set(NEW_RELATIONS)


def test_chunking_ignores_new_kinds_and_relations(
    corpus: ParsedCorpus, legacy_corpus: ParsedCorpus, counters: TokenCounters
) -> None:
    """The 6.1.0 additions never become searchable passages."""
    config = AppConfig()
    chunks = build_source_chunks(
        corpus.entities, corpus.edges, counters.embedding, config.chunking
    )
    legacy = build_source_chunks(
        legacy_corpus.entities,
        legacy_corpus.edges,
        counters.embedding,
        config.chunking,
    )
    assert [(chunk.owner_kind, chunk.owner_id, chunk.section) for chunk in chunks] == [
        (chunk.owner_kind, chunk.owner_id, chunk.section) for chunk in legacy
    ]
    assert all(
        chunk.owner_kind in {"technique", "mitigation", "case_study"}
        for chunk in chunks
    )
    assert all(chunk.section not in NEW_RELATIONS for chunk in chunks)


@pytest.mark.parametrize("identifier", ["AML.S0001", "AML.E0001"])
def test_new_kinds_are_never_graph_seeds(corpus: ParsedCorpus, identifier: str) -> None:
    """An explicit new-kind identifier does not start expansion."""
    graph = build_graph(corpus.entities, corpus.edges)
    assert extract_exact_ids(identifier) == (identifier,)
    assert (
        select_seeds(
            graph,
            fused=(),
            passage_owners={},
            anchors=(identifier,),
            limit=8,
        )
        == ()
    )


def test_traversal_never_follows_new_relations(corpus: ParsedCorpus) -> None:
    """Expansion keeps the original typed steps only."""
    graph = build_graph(corpus.entities, corpus.edges)
    seeds = select_seeds(
        graph,
        fused=(),
        passage_owners={},
        anchors=("AML.T0051.001",),
        limit=8,
    )
    traversal = traverse(graph, seeds, AppConfig().retrieval)
    relations = {hop.relation for path in traversal.paths for hop in path.hops}
    assert relations
    assert relations <= SEARCHABLE_RELATIONS
