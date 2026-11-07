"""Graph construction, seed selection, and bounded typed traversal."""

from __future__ import annotations

import pytest

from atlas_review.config import RetrievalConfig
from atlas_review.contracts import (
    AtlasEntity,
    AtlasRelationship,
    Candidate,
    EntityKind,
    GraphPath,
    GraphSeed,
    RelationName,
)
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.graph_index import AtlasGraph, build_graph
from atlas_review.retrieval.graph_search import (
    MAX_PATHS_PER_RESULT,
    Traversal,
    extract_exact_ids,
    owner_entities,
    select_seeds,
    traverse,
)

TECHNIQUE = "AML.T0051.001"
PARENT = "AML.T0051"
CASE = "AML.CS0035"
OTHER_CASE = "AML.CS0024"
MITIGATION = "AML.M0033"
TACTIC = "AML.TA0005"
OTHER_TECHNIQUE = "AML.T0012"


@pytest.fixture
def atlas_graph(corpus: ParsedCorpus) -> AtlasGraph:
    """Return the fixture corpus as a directed multigraph."""
    return build_graph(corpus.entities, corpus.edges)


def _edges_between(
    graph: AtlasGraph, source: str, target: str, relation: str
) -> tuple[AtlasRelationship, ...]:
    """Return every stored edge with the given endpoints and relation."""
    return graph.edges_between(source, target, relation)


def _reached(traversal: Traversal) -> set[str]:
    """Return the entity IDs reached by one traversal."""
    return set(traversal.node_hops)


def _path_to(traversal: Traversal, entity_id: str) -> GraphPath:
    """Return the best recorded path for one reached non-seed entity."""
    return traversal.node_paths[entity_id][0]


def _fused(*channels: str, passage_id: int = 1, rank: int = 1) -> Candidate:
    """Build one fused candidate with the given channel provenance."""
    return Candidate.model_validate(
        {
            "passage_id": passage_id,
            "channels": list(channels),
            "dense_rank": rank if "dense" in channels else None,
            "dense_distance": 0.5 if "dense" in channels else None,
            "bm25_rank": rank if "bm25" in channels else None,
            "bm25_score": 1.0 if "bm25" in channels else None,
            "fusion_score": 0.01,
        },
        strict=True,
    )


def _entity(entity_id: str, kind: EntityKind) -> AtlasEntity:
    """Build a minimal validated entity for synthetic graph tests."""
    return AtlasEntity(
        generation_id="gen",
        entity_id=entity_id,
        kind=kind,
        name=entity_id,
        display_name=entity_id,
        description="description",
        source_locator=f"/{kind}s/{entity_id}/description",
    )


def _edge(
    edge_id: str, source: str, target: str, relation: RelationName
) -> AtlasRelationship:
    """Build a minimal validated relationship for synthetic graph tests."""
    return AtlasRelationship(
        generation_id="gen",
        edge_id=edge_id,
        relation=relation,
        source_id=source,
        target_id=target,
        source_locator=f"/relationships/{source}/{relation}",
    )


def _config(**change: int) -> RetrievalConfig:
    """Return the default retrieval settings with explicit overrides."""
    return RetrievalConfig.model_validate(
        {**RetrievalConfig().model_dump(mode="python"), **change}, strict=False
    )


def _seeds(graph: AtlasGraph, *anchors: str) -> tuple[GraphSeed, ...]:
    """Return the seeds for the given exact anchors and no fused candidates."""
    return select_seeds(graph, fused=(), passage_owners={}, anchors=anchors, limit=8)


class TestGraphConstruction:
    """The stored records become a multigraph with orientation preserved."""

    def test_parallel_case_steps_survive_construction(
        self, atlas_graph: AtlasGraph
    ) -> None:
        parallel = _edges_between(atlas_graph, CASE, TECHNIQUE, "employs")
        assert len(parallel) == 2
        assert {edge.step_id for edge in parallel} == {"S01", "S03"}

    def test_edge_payload_keeps_original_orientation(
        self, atlas_graph: AtlasGraph
    ) -> None:
        for edge in atlas_graph.edges.values():
            index = atlas_graph.edges_by_id[edge.edge_id]
            source, target = atlas_graph.graph.get_edge_endpoints_by_index(index)
            assert atlas_graph.graph[source].entity_id == edge.source_id
            assert atlas_graph.graph[target].entity_id == edge.target_id

    def test_node_payloads_are_the_stored_entities(
        self, atlas_graph: AtlasGraph
    ) -> None:
        assert atlas_graph.entity(CASE).case_type == "Exercise"
        assert atlas_graph.entity(OTHER_CASE).case_type == "Incident"

    def test_graph_construction_is_deterministic(self, corpus: ParsedCorpus) -> None:
        first = build_graph(corpus.entities, corpus.edges)
        second = build_graph(
            tuple(reversed(corpus.entities)), tuple(reversed(corpus.edges))
        )
        assert first.node_by_id == second.node_by_id
        assert first.edges_by_id == second.edges_by_id

    def test_dangling_edge_endpoint_is_rejected(self) -> None:
        entities = [_entity("AML.T0001", "technique")]
        edges = [_edge("a" * 64, "AML.T0001", "AML.T0404", "employs")]
        with pytest.raises(AtlasReviewError) as caught:
            build_graph(entities, edges)
        assert caught.value.code == "dangling_edge"


class TestExactIdentifiers:
    """Explicit ATLAS identifiers in the focus become intentional anchors."""

    @pytest.mark.parametrize(
        ("focus", "expected"),
        [
            pytest.param("compare a CS0035 match", (), id="no-identifier"),
            pytest.param("why AML.CS0035 matters", ("AML.CS0035",), id="case"),
            pytest.param(
                "aml.cs0035 and aml.m0033 together",
                ("AML.CS0035", "AML.M0033"),
                id="lowercase",
            ),
            pytest.param(
                "AML.T0051.001 versus AML.T0051",
                ("AML.T0051.001", "AML.T0051"),
                id="parent-and-child",
            ),
            pytest.param(
                "AML.T0051 then AML.T0051 again",
                ("AML.T0051",),
                id="deduplicated",
            ),
        ],
    )
    def test_identifiers_are_extracted_in_order(
        self, focus: str, expected: tuple[str, ...]
    ) -> None:
        assert extract_exact_ids(focus) == expected

    def test_more_than_four_identifiers_are_rejected(self) -> None:
        focus = "AML.T0001 AML.T0002 AML.T0003 AML.T0004 AML.T0005"
        with pytest.raises(AtlasReviewError) as caught:
            extract_exact_ids(focus)
        assert caught.value.code == "too_many_explicit_ids"

    def test_unknown_identifier_is_rejected(self, atlas_graph: AtlasGraph) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            _seeds(atlas_graph, "AML.T9999")
        assert caught.value.code == "unknown_explicit_id"

    def test_tactic_identifier_is_resolvable_but_never_a_seed(
        self, atlas_graph: AtlasGraph
    ) -> None:
        assert _seeds(atlas_graph, TACTIC) == ()


class TestSeedSelection:
    """Exact anchors precede fused owners, and edge owners prefer the technique."""

    def test_exact_anchor_seeds_come_first(self, atlas_graph: AtlasGraph) -> None:
        seeds = select_seeds(
            atlas_graph,
            fused=(_fused("dense", passage_id=3),),
            passage_owners={3: (OTHER_TECHNIQUE,)},
            anchors=(CASE,),
            limit=8,
        )
        assert [seed.entity_id for seed in seeds] == [CASE, OTHER_TECHNIQUE]
        assert [seed.origin for seed in seeds] == ["exact_id", "fused"]

    def test_bm25_only_candidate_seeds_graph_search(
        self, atlas_graph: AtlasGraph
    ) -> None:
        seeds = select_seeds(
            atlas_graph,
            fused=(_fused("bm25", passage_id=9),),
            passage_owners={9: (OTHER_TECHNIQUE,)},
            anchors=(),
            limit=8,
        )
        assert [seed.entity_id for seed in seeds] == [OTHER_TECHNIQUE]
        assert seeds[0].channels == ["bm25"]
        assert seeds[0].fused_rank == 1

    def test_seed_limit_stops_new_entities(self, atlas_graph: AtlasGraph) -> None:
        fused = tuple(
            _fused("dense", passage_id=index, rank=index) for index in range(1, 6)
        )
        owners = {
            1: (PARENT,),
            2: (OTHER_TECHNIQUE,),
            3: (CASE,),
            4: (MITIGATION,),
            5: ("AML.TA0004",),
        }
        seeds = select_seeds(
            atlas_graph,
            fused=fused,
            passage_owners=owners,
            anchors=(),
            limit=3,
        )
        assert [seed.entity_id for seed in seeds] == [PARENT, OTHER_TECHNIQUE, CASE]

    def test_repeated_owner_seeds_once(self, atlas_graph: AtlasGraph) -> None:
        seeds = select_seeds(
            atlas_graph,
            fused=(
                _fused("dense", passage_id=1, rank=1),
                _fused("bm25", passage_id=2, rank=1),
            ),
            passage_owners={1: (PARENT,), 2: (PARENT,)},
            anchors=(PARENT,),
            limit=8,
        )
        assert [seed.entity_id for seed in seeds] == [PARENT]

    def test_edge_owner_contributes_technique_then_case(
        self, atlas_graph: AtlasGraph
    ) -> None:
        employs = _edges_between(atlas_graph, CASE, TECHNIQUE, "employs")[0]
        mitigates = _edges_between(atlas_graph, MITIGATION, PARENT, "mitigates")[0]
        assert owner_entities(employs, atlas_graph) == (TECHNIQUE, CASE)
        assert owner_entities(mitigates, atlas_graph) == (PARENT, MITIGATION)


class TestTraversal:
    """Only the documented typed steps expand, within the global caps."""

    def test_technique_reaches_cases_parents_and_mitigations(
        self, atlas_graph: AtlasGraph
    ) -> None:
        traversal = traverse(
            atlas_graph, _seeds(atlas_graph, TECHNIQUE), RetrievalConfig()
        )
        assert _reached(traversal) == {
            TECHNIQUE,
            CASE,
            PARENT,
            MITIGATION,
            OTHER_TECHNIQUE,
        }

    def test_tactic_hubs_are_never_traversed(self, atlas_graph: AtlasGraph) -> None:
        traversal = traverse(
            atlas_graph, _seeds(atlas_graph, TECHNIQUE), RetrievalConfig()
        )
        assert TACTIC not in _reached(traversal)
        hops = [hop for path in traversal.paths for hop in path.hops]
        assert all(
            hop.relation in {"employs", "mitigates", "specializes"} for hop in hops
        )

    def test_parent_mitigation_path_preserves_original_orientation(
        self, atlas_graph: AtlasGraph
    ) -> None:
        traversal = traverse(
            atlas_graph, _seeds(atlas_graph, TECHNIQUE), RetrievalConfig()
        )
        path = _path_to(traversal, MITIGATION)
        assert [hop.relation for hop in path.hops] == ["specializes", "mitigates"]
        assert [(hop.source_id, hop.target_id) for hop in path.hops] == [
            (TECHNIQUE, PARENT),
            (MITIGATION, PARENT),
        ]
        assert [hop.traversed_inverse for hop in path.hops] == [False, True]

    def test_cycle_terminates_with_bounded_paths_per_result(
        self, atlas_graph: AtlasGraph
    ) -> None:
        traversal = traverse(atlas_graph, _seeds(atlas_graph, CASE), RetrievalConfig())
        assert TECHNIQUE in _reached(traversal)
        assert max(traversal.node_hops.values()) <= 2
        assert set(traversal.node_paths) | {CASE} == _reached(traversal)
        for paths in traversal.node_paths.values():
            assert len(paths) <= MAX_PATHS_PER_RESULT
            assert len({path.seed_entity_id for path in paths}) == len(paths)

    def test_one_seed_produces_one_path_per_result(
        self, atlas_graph: AtlasGraph
    ) -> None:
        traversal = traverse(atlas_graph, _seeds(atlas_graph, CASE), RetrievalConfig())
        assert all(len(paths) == 1 for paths in traversal.node_paths.values())

    def test_two_seeds_reaching_one_result_keep_best_path_per_seed(
        self, atlas_graph: AtlasGraph
    ) -> None:
        traversal = traverse(
            atlas_graph, _seeds(atlas_graph, TECHNIQUE, MITIGATION), RetrievalConfig()
        )
        paths = traversal.node_paths[PARENT]
        assert [path.seed_entity_id for path in paths] == [TECHNIQUE, MITIGATION]
        assert [hop.relation for hop in paths[0].hops] == ["specializes"]
        assert [hop.relation for hop in paths[1].hops] == ["mitigates"]

    def test_traversed_edges_are_recorded_when_all_results_are_seeds(
        self, atlas_graph: AtlasGraph
    ) -> None:
        traversal = traverse(
            atlas_graph,
            _seeds(
                atlas_graph,
                CASE,
                TECHNIQUE,
                PARENT,
                MITIGATION,
                OTHER_TECHNIQUE,
                OTHER_CASE,
            ),
            RetrievalConfig(),
        )
        assert set(traversal.node_hops) <= {seed.entity_id for seed in traversal.seeds}
        assert traversal.edge_paths
        assert traversal.paths

    @pytest.mark.parametrize(
        ("change", "expected_code"),
        [
            pytest.param({"graph_max_hops": 0}, None, id="zero-hops"),
            pytest.param({"graph_max_nodes": 1}, "graph_max_nodes", id="node-cap"),
            pytest.param({"graph_max_edges": 1}, "graph_max_edges", id="edge-cap"),
            pytest.param(
                {"graph_max_edges_per_node": 1},
                "graph_max_edges_per_node",
                id="fan-out-cap",
            ),
        ],
    )
    def test_caps_bound_the_expansion(
        self,
        atlas_graph: AtlasGraph,
        change: dict[str, int],
        expected_code: str | None,
    ) -> None:
        traversal = traverse(
            atlas_graph, _seeds(atlas_graph, TECHNIQUE), _config(**change)
        )
        if expected_code is None:
            assert traversal.paths == ()
            assert _reached(traversal) == {TECHNIQUE}
            return
        assert expected_code in traversal.limits_reached

    def test_hop_limit_stops_after_two_hops(self, atlas_graph: AtlasGraph) -> None:
        traversal = traverse(
            atlas_graph, _seeds(atlas_graph, TECHNIQUE), RetrievalConfig()
        )
        assert max(traversal.node_hops.values()) <= 2

    def test_fan_out_slots_are_round_robin_across_categories(
        self, atlas_graph: AtlasGraph
    ) -> None:
        seeds = _seeds(atlas_graph, TECHNIQUE)
        first_slot = {
            path.entity_id
            for path in traverse(
                atlas_graph, seeds, _config(graph_max_edges_per_node=1)
            ).paths
        }
        two_slots = [
            path.entity_id
            for path in traverse(
                atlas_graph, seeds, _config(graph_max_edges_per_node=2)
            ).paths
        ]
        assert first_slot == {CASE}
        assert two_slots[:2] == [CASE, PARENT]

    def test_traversal_is_repeatable(self, atlas_graph: AtlasGraph) -> None:
        seeds = _seeds(atlas_graph, CASE)
        first = traverse(atlas_graph, seeds, RetrievalConfig())
        second = traverse(atlas_graph, seeds, RetrievalConfig())
        assert first.paths == second.paths
        assert first.node_hops == second.node_hops

    def test_no_seed_produces_no_paths(self, atlas_graph: AtlasGraph) -> None:
        traversal = traverse(atlas_graph, (), RetrievalConfig())
        assert traversal.paths == ()
        assert traversal.node_hops == {}
