"""Graph seeding, bounded typed traversal, and graph-derived passage choice.

Seeds come from explicit ATLAS identifiers and from the fused direct ranking.
Only the documented typed steps expand; matrix order (`sequences`) and tactic
association (`achieves`) never create candidates.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from atlas_review.config import RetrievalConfig
from atlas_review.contracts import (
    AtlasRelationship,
    Candidate,
    EntityKind,
    GraphHop,
    GraphPath,
    GraphSeed,
    Passage,
)
from atlas_review.errors import STAGE_RETRIEVAL, AtlasReviewError
from atlas_review.retrieval.graph_index import AtlasGraph

DEFAULT_EXACT_ID_LIMIT = 4
MAX_PATHS_PER_RESULT = 2
SEEDABLE_KINDS = frozenset({"technique", "case_study", "mitigation"})
_CATEGORY_RANK = {
    "technique": 0,
    "case_study": 1,
    "mitigation": 2,
    "relationship": 3,
}
_ATLAS_ID = re.compile(r"\bAML\.(?:TA|CS|T|M)\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

# Allowed step names per current node kind, in the round-robin order.
_ALLOWED_STEPS: dict[str, tuple[str, ...]] = {
    "technique": (
        "employs_in",
        "mitigates_in",
        "specializes_out",
        "specializes_in",
    ),
    "case_study": ("employs_out",),
    "mitigation": ("mitigates_out",),
}
_RELATION_BY_STEP = {
    "employs_in": ("employs", True),
    "employs_out": ("employs", False),
    "mitigates_in": ("mitigates", True),
    "mitigates_out": ("mitigates", False),
    "specializes_in": ("specializes", True),
    "specializes_out": ("specializes", False),
}


@dataclass(frozen=True, slots=True)
class Step:
    """One eligible traversal step from the current entity.

    Instance Attributes:
        edge (AtlasRelationship): Stored edge that would be traversed.
        inverse (bool): The step follows the edge against its stored direction.
    """

    edge: AtlasRelationship
    inverse: bool


@dataclass(frozen=True, slots=True)
class Traversal:
    """Result of one bounded breadth-first expansion.

    Instance Attributes:
        seeds (tuple[GraphSeed, ...]): Seeds the expansion started from.
        paths (tuple[GraphPath, ...]): Every discovered provenance path.
        node_paths (dict[str, tuple[GraphPath, ...]]): Best paths per reached
            entity, at most one per originating seed and two in total.
        edge_paths (dict[str, GraphPath]): Path per traversed edge.
        node_hops (dict[str, int]): Shortest hop count per reached entity.
        node_kinds (dict[str, EntityKind]): Entity kind per reached entity.
        seed_rank (dict[str, int]): Seed position per seed entity.
        limits_reached (tuple[str, ...]): Caps that stopped part of the search.
    """

    seeds: tuple[GraphSeed, ...]
    paths: tuple[GraphPath, ...]
    node_paths: dict[str, tuple[GraphPath, ...]]
    edge_paths: dict[str, GraphPath]
    node_hops: dict[str, int]
    node_kinds: dict[str, EntityKind]
    seed_rank: dict[str, int]
    limits_reached: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphCandidate:
    """One passage introduced by graph expansion.

    Instance Attributes:
        passage_id (int): Stored passage identifier.
        owner_id (str): Entity or edge identifier that owns the passage.
        category (str): Owner category used for the round-robin order.
        hops (int): Path length from the originating seed.
        seed_rank (int): Position of the originating seed.
        paths (tuple[GraphPath, ...]): Recorded paths to the owner, best first.
    """

    passage_id: int
    owner_id: str
    category: str
    hops: int
    seed_rank: int
    paths: tuple[GraphPath, ...]


@dataclass(frozen=True, slots=True)
class GraphExpansion:
    """Seeds, paths, and graph-derived passages for one retrieval run.

    Instance Attributes:
        seeds (tuple[GraphSeed, ...]): Seeds in priority order.
        paths (tuple[GraphPath, ...]): Reached node and edge paths.
        candidates (tuple[GraphCandidate, ...]): Graph-derived passages.
        anchors (tuple[str, ...]): Exact identifiers found in the focus.
        limits_reached (tuple[str, ...]): Caps that stopped part of the search.
    """

    seeds: tuple[GraphSeed, ...]
    paths: tuple[GraphPath, ...]
    candidates: tuple[GraphCandidate, ...]
    anchors: tuple[str, ...]
    limits_reached: tuple[str, ...]


def extract_exact_ids(
    focus: str, limit: int = DEFAULT_EXACT_ID_LIMIT
) -> tuple[str, ...]:
    """Return explicit ATLAS identifiers in the focus, in first-seen order.

    Args:
        focus (str): Ordinary retrieval focus text.
        limit (int): Maximum accepted distinct identifiers.

    Returns:
        (tuple[str, ...]) Upper-cased distinct identifiers.

    Raises:
        AtlasReviewError: The focus names more identifiers than the limit.
    """
    identifiers: list[str] = []
    for match in _ATLAS_ID.finditer(focus):
        identifier = match.group(0).upper()
        if identifier not in identifiers:
            identifiers.append(identifier)
    if len(identifiers) > limit:
        raise _retrieval_error(
            "too_many_explicit_ids",
            f"the focus names {len(identifiers)} ATLAS identifiers; "
            f"at most {limit} can anchor retrieval",
        )
    return tuple(identifiers)


def owner_entities(edge: AtlasRelationship, graph: AtlasGraph) -> tuple[str, ...]:
    """Return the seedable entities of an edge-owned passage, technique first.

    Args:
        edge (AtlasRelationship): Edge owning the passage.
        graph (AtlasGraph): Graph holding both endpoints.

    Returns:
        (tuple[str, ...]) Technique endpoint first, then the other endpoint.

    Raises:
        AtlasReviewError: An endpoint is missing from the graph.
    """
    endpoints = (graph.entity(edge.source_id), graph.entity(edge.target_id))
    techniques = [item.entity_id for item in endpoints if item.kind == "technique"]
    others = [
        item.entity_id
        for item in endpoints
        if item.kind in SEEDABLE_KINDS and item.kind != "technique"
    ]
    return tuple(techniques + others)


def select_seeds(
    graph: AtlasGraph,
    *,
    fused: Sequence[Candidate],
    passage_owners: Mapping[int, tuple[str, ...]],
    anchors: Sequence[str],
    limit: int,
) -> tuple[GraphSeed, ...]:
    """Select expansion seeds from exact anchors and the fused ranking.

    Args:
        graph (AtlasGraph): Generation graph.
        fused (Sequence[Candidate]): Fused direct candidates, best first.
        passage_owners (Mapping[int, tuple[str, ...]]): Seedable owners per
            passage, in contribution order.
        anchors (Sequence[str]): Explicit ATLAS identifiers from the focus.
        limit (int): Maximum number of distinct seeds.

    Returns:
        (tuple[GraphSeed, ...]) Seeds in priority order.

    Raises:
        AtlasReviewError: An explicit identifier is unknown, or a fused owner
            is missing from the graph.
    """
    seeds: list[GraphSeed] = []
    seen: set[str] = set()

    def admit(
        entity_id: str,
        origin: str,
        fused_rank: int | None,
        channels: Sequence[str],
    ) -> None:
        if entity_id in seen or len(seeds) >= limit:
            return
        if graph.entity(entity_id).kind not in SEEDABLE_KINDS:
            return
        seen.add(entity_id)
        seeds.append(
            GraphSeed.model_validate(
                {
                    "entity_id": entity_id,
                    "origin": origin,
                    "fused_rank": fused_rank,
                    "channels": list(channels),
                },
                strict=True,
            )
        )

    for identifier in anchors:
        try:
            graph.entity(identifier)
        except AtlasReviewError as exc:
            raise _retrieval_error(
                "unknown_explicit_id",
                f"the focus names {identifier}, which this generation does not contain",
            ) from exc
        admit(identifier, "exact_id", None, ("exact_id",))
    for rank, candidate in enumerate(fused, start=1):
        for entity_id in passage_owners.get(candidate.passage_id, ()):
            admit(entity_id, "fused", rank, candidate.channels)
    return tuple(seeds)


def traverse(
    graph: AtlasGraph, seeds: Sequence[GraphSeed], config: RetrievalConfig
) -> Traversal:
    """Expand from the seeds over the allowed typed steps.

    Args:
        graph (AtlasGraph): Generation graph.
        seeds (Sequence[GraphSeed]): Seeds in priority order.
        config (RetrievalConfig): Hop, node, edge, and fan-out caps.

    Returns:
        (Traversal) Reached entities, paths, and any caps that were reached.
    """
    if not seeds:
        return Traversal((), (), {}, {}, {}, {}, {}, ())
    limits: set[str] = set()
    seed_rank = {seed.entity_id: index for index, seed in enumerate(seeds)}
    node_hops = dict.fromkeys(seed_rank, 0)
    node_kinds: dict[str, EntityKind] = {
        entity_id: graph.entity(entity_id).kind for entity_id in seed_rank
    }
    node_paths: dict[str, list[GraphPath]] = {}
    edge_paths: dict[str, GraphPath] = {}
    visited_edges: set[str] = set()
    queue: deque[tuple[str, str, tuple[GraphHop, ...]]] = deque(
        (seed.entity_id, seed.entity_id, ()) for seed in seeds
    )
    while queue:
        origin, entity_id, hops = queue.popleft()
        if len(hops) >= config.graph_max_hops:
            continue
        steps = _eligible_steps(graph, entity_id)
        if sum(len(items) for items in steps.values()) > (
            config.graph_max_edges_per_node
        ):
            limits.add("graph_max_edges_per_node")
        for step in _round_robin(steps, config.graph_max_edges_per_node):
            if step.edge.edge_id in visited_edges:
                continue
            if len(visited_edges) >= config.graph_max_edges:
                limits.add("graph_max_edges")
                break
            neighbour = step.edge.source_id if step.inverse else step.edge.target_id
            if neighbour not in node_hops and len(node_hops) >= config.graph_max_nodes:
                limits.add("graph_max_nodes")
                continue
            visited_edges.add(step.edge.edge_id)
            path = GraphPath(
                seed_entity_id=origin,
                entity_id=neighbour,
                hops=[
                    *hops,
                    GraphHop(
                        edge_id=step.edge.edge_id,
                        source_id=step.edge.source_id,
                        target_id=step.edge.target_id,
                        relation=step.edge.relation,
                        traversed_inverse=step.inverse,
                    ),
                ],
            )
            edge_paths.setdefault(step.edge.edge_id, path)
            recorded = node_paths.setdefault(neighbour, [])
            if len(recorded) < MAX_PATHS_PER_RESULT and all(
                known.seed_entity_id != origin for known in recorded
            ):
                recorded.append(path)
            if neighbour in node_hops:
                continue
            node_hops[neighbour] = len(path.hops)
            node_kinds[neighbour] = graph.entity(neighbour).kind
            queue.append((origin, neighbour, tuple(path.hops)))
    reached = {
        entity_id: tuple(
            sorted(
                paths_per_node,
                key=lambda path: (
                    seed_rank[path.seed_entity_id],
                    len(path.hops),
                ),
            )
        )
        for entity_id, paths_per_node in node_paths.items()
    }
    return Traversal(
        seeds=tuple(seeds),
        paths=_recorded_paths(reached, edge_paths, seed_rank),
        node_paths=reached,
        edge_paths=edge_paths,
        node_hops=node_hops,
        node_kinds=node_kinds,
        seed_rank=seed_rank,
        limits_reached=tuple(sorted(limits)),
    )


def _recorded_paths(
    node_paths: Mapping[str, Sequence[GraphPath]],
    edge_paths: Mapping[str, GraphPath],
    seed_rank: Mapping[str, int],
) -> tuple[GraphPath, ...]:
    """Return every discovered path once, best first, for the trace."""
    distinct: dict[tuple[str, str, tuple[str, ...]], GraphPath] = {}
    for paths_per_node in node_paths.values():
        for path in paths_per_node:
            distinct.setdefault(_path_key(path), path)
    for edge_id in sorted(edge_paths):
        path = edge_paths[edge_id]
        distinct.setdefault(_path_key(path), path)
    return tuple(
        sorted(
            distinct.values(),
            key=lambda path: (
                seed_rank[path.seed_entity_id],
                len(path.hops),
                path.entity_id,
                tuple(hop.edge_id for hop in path.hops),
            ),
        )
    )


def _path_key(path: GraphPath) -> tuple[str, str, tuple[str, ...]]:
    """Return the identity of one provenance path."""
    return (
        path.seed_entity_id,
        path.entity_id,
        tuple(hop.edge_id for hop in path.hops),
    )


def graph_passages(
    traversal: Traversal,
    passages: Mapping[int, Passage],
    limit: int,
) -> tuple[GraphCandidate, ...]:
    """Choose graph-derived passages in the documented deterministic order.

    Args:
        traversal (Traversal): Result of one bounded expansion.
        passages (Mapping[int, Passage]): Stored passages of the generation.
        limit (int): Maximum number of graph-derived passages.

    Returns:
        (tuple[GraphCandidate, ...]) Passages in breadth-first fair order.
    """
    by_entity: dict[str, list[Passage]] = {}
    by_edge: dict[str, list[Passage]] = {}
    for passage in passages.values():
        if passage.entity_id is not None:
            by_entity.setdefault(passage.entity_id, []).append(passage)
        elif passage.edge_id is not None:
            by_edge.setdefault(passage.edge_id, []).append(passage)
    owners = [
        *_entity_owners(traversal, by_entity),
        *_edge_owners(traversal, by_edge),
    ]
    ordered = sorted(
        owners,
        key=lambda owner: (
            owner[0],
            owner[1],
            _CATEGORY_RANK[owner[2]],
            owner[3],
        ),
    )
    chosen: list[GraphCandidate] = []
    for per_owner in range(2):
        for seed_rank, hops, category, owner_id, paths, owned in ordered:
            if per_owner >= len(owned):
                continue
            chosen.append(
                GraphCandidate(
                    passage_id=owned[per_owner].passage_id,
                    owner_id=owner_id,
                    category=category,
                    hops=hops,
                    seed_rank=seed_rank,
                    paths=paths,
                )
            )
            if len(chosen) >= limit:
                return tuple(chosen)
    return tuple(chosen)


def expand_graph(
    graph: AtlasGraph,
    passages: Mapping[int, Passage],
    fused: Sequence[Candidate],
    focus: str,
    config: RetrievalConfig,
) -> GraphExpansion:
    """Seed, expand, and choose graph-derived passages for one run.

    Args:
        graph (AtlasGraph): Generation graph.
        passages (Mapping[int, Passage]): Stored passages of the generation.
        fused (Sequence[Candidate]): Fused direct candidates, best first.
        focus (str): Ordinary retrieval focus.
        config (RetrievalConfig): Retrieval caps.

    Returns:
        (GraphExpansion) Seeds, discovered paths, and graph passages.

    Raises:
        AtlasReviewError: An explicit identifier is unknown or over the limit.
    """
    anchors = extract_exact_ids(focus, config.exact_id_limit)
    passage_owners = {
        passage_id: (
            (passage.entity_id,)
            if passage.entity_id is not None
            else owner_entities(graph.edge(passage.edge_id or ""), graph)
        )
        for passage_id, passage in passages.items()
    }
    seeds = select_seeds(
        graph,
        fused=fused,
        passage_owners=passage_owners,
        anchors=anchors,
        limit=config.seed_entity_limit,
    )
    traversal = traverse(graph, seeds, config)
    return GraphExpansion(
        seeds=seeds,
        paths=traversal.paths,
        candidates=graph_passages(traversal, passages, config.graph_passage_limit),
        anchors=anchors,
        limits_reached=traversal.limits_reached,
    )


def _eligible_steps(graph: AtlasGraph, entity_id: str) -> dict[str, list[Step]]:
    """Return the allowed steps of one entity grouped by step name."""
    node = graph.node_by_id[entity_id]
    outgoing = sorted(graph.graph.out_edge_indices(node))
    incoming = sorted(graph.graph.in_edge_indices(node))
    candidates = {
        "employs_in": tuple(graph.edge_at(index) for index in incoming),
        "employs_out": tuple(graph.edge_at(index) for index in outgoing),
        "mitigates_in": tuple(graph.edge_at(index) for index in incoming),
        "mitigates_out": tuple(graph.edge_at(index) for index in outgoing),
        "specializes_in": tuple(graph.edge_at(index) for index in incoming),
        "specializes_out": tuple(graph.edge_at(index) for index in outgoing),
    }
    kind = graph.entity(entity_id).kind
    steps: dict[str, list[Step]] = {}
    for name in _ALLOWED_STEPS.get(kind, ()):
        relation, inverse = _RELATION_BY_STEP[name]
        steps[name] = sorted(
            (
                Step(edge=edge, inverse=inverse)
                for edge in candidates[name]
                if edge.relation == relation
            ),
            key=lambda step: step.edge.edge_id,
        )
    return steps


def _round_robin(
    steps: Mapping[str, Sequence[Step]], allowance: int
) -> tuple[Step, ...]:
    """Take edge slots round-robin across categories, stably within each."""
    queues = [list(items) for items in steps.values()]
    taken: list[Step] = []
    while len(taken) < allowance and any(queues):
        for queue in queues:
            if queue and len(taken) < allowance:
                taken.append(queue.pop(0))
    return tuple(taken)


def _entity_owners(
    traversal: Traversal, by_entity: Mapping[str, Sequence[Passage]]
) -> list[tuple[int, int, str, str, tuple[GraphPath, ...], Sequence[Passage]]]:
    """Return the ordering keys for reached entity-owned passages."""
    owners: list[
        tuple[int, int, str, str, tuple[GraphPath, ...], Sequence[Passage]]
    ] = []
    for entity_id, hops in traversal.node_hops.items():
        owned = by_entity.get(entity_id)
        if not owned:
            continue
        ordered_passages = tuple(
            sorted(owned, key=lambda item: (item.ordinal, item.passage_id))
        )
        own_rank = traversal.seed_rank.get(entity_id)
        if own_rank is not None:
            owners.append(
                (
                    own_rank,
                    0,
                    traversal.node_kinds[entity_id],
                    entity_id,
                    (),
                    ordered_passages,
                )
            )
            continue
        paths = traversal.node_paths.get(entity_id, ())
        best = paths[0]
        owners.append(
            (
                traversal.seed_rank[best.seed_entity_id],
                hops,
                traversal.node_kinds[entity_id],
                entity_id,
                paths,
                ordered_passages,
            )
        )
    return owners


def _edge_owners(
    traversal: Traversal, by_edge: Mapping[str, Sequence[Passage]]
) -> list[tuple[int, int, str, str, tuple[GraphPath, ...], Sequence[Passage]]]:
    """Return the ordering keys for traversed relationship description passages."""
    owners: list[
        tuple[int, int, str, str, tuple[GraphPath, ...], Sequence[Passage]]
    ] = []
    for edge_id, path in traversal.edge_paths.items():
        owned = by_edge.get(edge_id)
        if not owned:
            continue
        owners.append(
            (
                traversal.seed_rank[path.seed_entity_id],
                len(path.hops),
                "relationship",
                edge_id,
                (path,),
                tuple(sorted(owned, key=lambda item: (item.ordinal, item.passage_id))),
            )
        )
    return owners


def _retrieval_error(code: str, message: str) -> AtlasReviewError:
    """Build a retrieval-input error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_RETRIEVAL,
        message=message,
        recovery=(
            "Remove the unsupported identifier from the retrieval focus, or "
            "rebuild the generation from the intended source."
        ),
    )
