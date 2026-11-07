"""Candidate-pool quotas and evidence closure over original-source passages.

The pool admits direct candidates, graph-derived passages, and a fair backfill
without letting one channel occupy every slot. Selection then accepts whole
evidence units, always with the connector cards needed to explain a path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Literal, cast

from atlas_review.config import AppConfig
from atlas_review.contracts import (
    AllowedAssociation,
    AtlasRelationship,
    Candidate,
    EvidenceChannel,
    EvidenceItem,
    EvidencePacket,
    GraphPath,
    OwnerKind,
    Passage,
    PassageProvenance,
    SearchChannel,
)
from atlas_review.retrieval.graph_index import AtlasGraph
from atlas_review.retrieval.graph_search import GraphCandidate, GraphExpansion

_CHANNEL_ORDER: tuple[SearchChannel, ...] = ("dense", "bm25", "exact_id", "graph")
_CONNECTOR: EvidenceChannel = "graph_connector"
_MISSING = (
    ("case_study", "No case-study narrative included"),
    ("mitigation", "No mitigation description included"),
    ("technique", "No technique description included"),
)
SourceKind = Literal["case_study", "mitigation"]
_OWNER_KINDS = frozenset({"technique", "mitigation", "case_study"})


@dataclass(frozen=True, slots=True)
class Pool:
    """The capped candidate pool passed to the reranker.

    Instance Attributes:
        candidates (tuple[Candidate, ...]): Admitted candidates in pool order.
        direct_count (int): Candidates admitted by the direct quota.
        graph_count (int): Candidates admitted by graph expansion.
        direct_quota_before_dedup (int): Passages offered to the direct quota.
        graph_quota_before_dedup (int): Graph passages offered to the pool.
        limits_reached (tuple[str, ...]): Caps that left candidates out.
    """

    candidates: tuple[Candidate, ...]
    direct_count: int
    graph_count: int
    direct_quota_before_dedup: int
    graph_quota_before_dedup: int
    limits_reached: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Selection:
    """Frozen evidence for one run, with exclusions and provenance.

    Instance Attributes:
        packet (EvidencePacket): Original-source evidence and allowlists.
        exclusions (tuple[str, ...]): Units that did not fit or were skipped.
        missing_categories (tuple[str, ...]): Evidence categories not covered.
        provenance (tuple[PassageProvenance, ...]): Hashes of used passages.
    """

    packet: EvidencePacket
    exclusions: tuple[str, ...]
    missing_categories: tuple[str, ...]
    provenance: tuple[PassageProvenance, ...]


@dataclass(slots=True)
class _Entry:
    """Mutable pool entry before it becomes a validated candidate."""

    passage_id: int
    channels: set[SearchChannel] = field(default_factory=set)
    dense_rank: int | None = None
    dense_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fusion_score: float = 0.0
    graph_paths: list[GraphPath] = field(default_factory=list)


@dataclass(slots=True)
class _Draft:
    """One evidence item before its review-local identifier is assigned."""

    item_kind: Literal["passage", "relationship"]
    entity_ids: tuple[str, ...]
    passage_id: int | None
    edge_id: str | None
    text: str
    source_locator: str
    case_type: str | None
    channels: tuple[EvidenceChannel, ...]
    graph_path: GraphPath | None


@dataclass(slots=True)
class _PoolState:
    """Mutable bookkeeping while the candidate pool is filled."""

    entries: dict[int, _Entry]
    order: list[int]
    quota: list[int]
    graph_ids: list[int]
    dropped: bool = False


@dataclass(frozen=True, slots=True)
class _Link:
    """One source relationship candidate for the association allowlist."""

    source_id: str
    source_kind: SourceKind
    technique_id: str
    parent: str | None
    edges: tuple[AtlasRelationship, ...]


def build_pool(
    fused: Sequence[Candidate],
    expansion: GraphExpansion,
    passages: Mapping[int, Passage],
    config: AppConfig,
) -> Pool:
    """Merge direct and graph candidates under the documented quota policy.

    Args:
        fused (Sequence[Candidate]): Fused direct candidates, best first.
        expansion (GraphExpansion): Graph-derived passages and anchor IDs.
        passages (Mapping[int, Passage]): Stored passages by identifier.
        config (AppConfig): Retrieval candidate budgets.

    Returns:
        (Pool) Capped pool with channel and path provenance attached.
    """
    limits = config.retrieval
    state = _PoolState(
        entries=_start_entries(fused, expansion), order=[], quota=[], graph_ids=[]
    )
    _add_anchors(state, expansion, passages)
    _add_direct_quota(state, fused, limits.direct_candidate_limit)
    _add_graph_candidates(state, expansion, limits.rerank_candidate_limit)
    _backfill(state, fused, expansion, limits.rerank_candidate_limit)
    limits_reached = expansion.limits_reached
    if state.dropped:
        limits_reached = (*limits_reached, "rerank_candidate_limit")
    return Pool(
        candidates=tuple(
            _candidate(state.entries[passage_id]) for passage_id in state.order
        ),
        direct_count=len(state.quota),
        graph_count=len(state.graph_ids),
        direct_quota_before_dedup=len(expansion.anchors) + len(fused),
        graph_quota_before_dedup=len(expansion.candidates),
        limits_reached=limits_reached,
    )


def _start_entries(
    fused: Sequence[Candidate], expansion: GraphExpansion
) -> dict[int, _Entry]:
    """Build one entry per fused candidate and attach known graph paths."""
    entries = {
        candidate.passage_id: _Entry(
            passage_id=candidate.passage_id,
            channels=set(candidate.channels),
            dense_rank=candidate.dense_rank,
            dense_distance=candidate.dense_distance,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            fusion_score=candidate.fusion_score,
        )
        for candidate in fused
    }
    for candidate in expansion.candidates:
        entry = entries.get(candidate.passage_id)
        if entry is not None and not entry.graph_paths:
            entry.graph_paths = list(candidate.paths)
    return entries


def _add_anchors(
    state: _PoolState,
    expansion: GraphExpansion,
    passages: Mapping[int, Passage],
) -> None:
    """Reserve the best passage of every explicit identifier anchor."""
    for anchor in expansion.anchors:
        passage_id = _first_passage_of(anchor, passages)
        if passage_id is None or passage_id in state.order:
            continue
        entry = state.entries.setdefault(passage_id, _Entry(passage_id=passage_id))
        entry.channels.add("exact_id")
        state.order.append(passage_id)


def _add_direct_quota(
    state: _PoolState, fused: Sequence[Candidate], direct_limit: int
) -> None:
    """Fill the direct quota from the fused ranking and the reserved anchors."""
    state.quota.extend(state.order)
    for candidate in fused:
        if len(state.quota) >= direct_limit:
            break
        if candidate.passage_id not in state.quota:
            state.quota.append(candidate.passage_id)
    state.order = list(state.quota)


def _add_graph_candidates(
    state: _PoolState, expansion: GraphExpansion, pool_limit: int
) -> None:
    """Fill remaining capacity with graph-derived passages."""
    for candidate in expansion.candidates:
        if candidate.passage_id in state.order:
            continue
        if len(state.order) >= pool_limit:
            state.dropped = True
            return
        state.entries[candidate.passage_id] = _graph_entry(candidate)
        state.order.append(candidate.passage_id)
        state.graph_ids.append(candidate.passage_id)


def _backfill(
    state: _PoolState,
    fused: Sequence[Candidate],
    expansion: GraphExpansion,
    pool_limit: int,
) -> None:
    """Alternate leftover direct and graph candidates while capacity remains."""
    admitted = set(state.order)
    remaining_fused = [
        candidate.passage_id
        for candidate in fused
        if candidate.passage_id not in admitted
    ]
    remaining_graph = [
        candidate.passage_id
        for candidate in expansion.candidates
        if candidate.passage_id not in admitted
    ]
    for fused_id, graph_id in zip_longest(remaining_fused, remaining_graph):
        for passage_id, from_graph in ((fused_id, False), (graph_id, True)):
            if passage_id is None or passage_id in admitted:
                continue
            if len(state.order) >= pool_limit:
                state.dropped = True
                return
            if passage_id not in state.entries:
                state.entries[passage_id] = (
                    _graph_entry(_graph_candidate(passage_id, expansion))
                    if from_graph
                    else _Entry(passage_id=passage_id)
                )
            if from_graph:
                state.graph_ids.append(passage_id)
            admitted.add(passage_id)
            state.order.append(passage_id)


def select_evidence(
    ranked: Sequence[Candidate],
    *,
    graph: AtlasGraph,
    passages: Mapping[int, Passage],
    generation_id: str,
    collection_version: str,
    config: AppConfig,
) -> Selection:
    """Select whole evidence units from the reranked candidates.

    Args:
        ranked (Sequence[Candidate]): Reranked pool, best first.
        graph (AtlasGraph): Generation graph used for connector cards.
        passages (Mapping[int, Passage]): Stored passages by identifier.
        generation_id (str): Generation the evidence belongs to.
        collection_version (str): ATLAS collection version of the generation.
        config (AppConfig): Evidence budgets.

    Returns:
        (Selection) Frozen packet, exclusions, gaps, and provenance.
    """
    limits = config.retrieval
    drafts: list[_Draft] = []
    cards: dict[str, str] = {}
    owners: dict[str, int] = {}
    exclusions: list[str] = []
    selected: list[int] = []
    characters = 0
    covered = {"case_study": False, "mitigation": False, "technique": False}
    for candidate in ranked:
        if len(selected) >= limits.selected_passage_limit:
            break
        passage = passages.get(candidate.passage_id)
        if passage is None:
            exclusions.append(f"p{candidate.passage_id}: no stored passage")
            continue
        owner_key = passage.entity_id or passage.edge_id or ""
        if owners.get(owner_key, 0) >= limits.per_owner_passage_limit:
            exclusions.append(f"p{candidate.passage_id}: per-owner passage limit")
            continue
        unit = _unit(passage, candidate, graph, cards)
        unit.extend(_connecting_cards(unit, drafts, graph))
        new = _new_drafts(unit, drafts)
        cost = sum(len(draft.text) for draft in new)
        if (
            len(drafts) + len(new) > limits.evidence_item_limit
            or characters + cost > limits.evidence_character_limit
        ):
            exclusions.append(f"p{candidate.passage_id}: evidence budget")
            continue
        for draft in new:
            drafts.append(draft)
            if draft.edge_id is not None and draft.passage_id is None:
                cards[draft.edge_id] = f"E{len(drafts):03d}"
        characters += cost
        owners[owner_key] = owners.get(owner_key, 0) + 1
        selected.append(passage.passage_id)
        _cover(covered, passage, graph)
    items = _items(drafts, collection_version)
    entity_items: dict[str, list[str]] = {}
    for item in items:
        for entity_id in item.entity_ids:
            entity_items.setdefault(entity_id, []).append(item.evidence_id)
    named = {entity for item in items for entity in item.entity_ids}
    return Selection(
        packet=EvidencePacket(
            schema_version=1,
            generation_id=generation_id,
            collection_version=collection_version,
            items=items,
            allowed_entity_ids_by_kind=_allowlist(graph, named),
            allowed_associations=_associations(
                graph=graph,
                cards=cards,
                entity_items=entity_items,
                named=named,
            ),
        ),
        exclusions=tuple(exclusions),
        missing_categories=tuple(
            message for kind, message in _MISSING if not covered[kind]
        ),
        provenance=tuple(
            PassageProvenance(
                passage_id=passage_id,
                source_sha256=passages[passage_id].source_sha256,
                context_cache_key=passages[passage_id].context_cache_key,
                content_sha256=passages[passage_id].content_sha256,
            )
            for passage_id in selected
        ),
    )


def _first_passage_of(entity_id: str, passages: Mapping[int, Passage]) -> int | None:
    """Return the first searchable passage of an entity anchor."""
    owned = sorted(
        (passage for passage in passages.values() if passage.entity_id == entity_id),
        key=lambda passage: (passage.ordinal, passage.passage_id),
    )
    return owned[0].passage_id if owned else None


def _graph_candidate(passage_id: int, expansion: GraphExpansion) -> GraphCandidate:
    """Return the recorded graph candidate for a passage identifier."""
    return next(
        candidate
        for candidate in expansion.candidates
        if candidate.passage_id == passage_id
    )


def _graph_entry(candidate: GraphCandidate) -> _Entry:
    """Build a pool entry admitted only by graph expansion."""
    return _Entry(
        passage_id=candidate.passage_id,
        channels={"graph"},
        graph_paths=list(candidate.paths),
    )


def _candidate(entry: _Entry) -> Candidate:
    """Validate one pool entry as a candidate."""
    return Candidate(
        passage_id=entry.passage_id,
        channels=[channel for channel in _CHANNEL_ORDER if channel in entry.channels],
        dense_rank=entry.dense_rank,
        dense_distance=entry.dense_distance,
        bm25_rank=entry.bm25_rank,
        bm25_score=entry.bm25_score,
        fusion_score=entry.fusion_score,
        graph_paths=list(entry.graph_paths),
        selected=False,
    )


def _unit(
    passage: Passage,
    candidate: Candidate,
    graph: AtlasGraph,
    cards: Mapping[str, str],
) -> list[_Draft]:
    """Build the complete evidence unit for one ranked passage."""
    unit = [_passage_draft(passage, candidate, graph)]
    if passage.edge_id is not None:
        unit.append(_card_draft(graph.edge(passage.edge_id), graph))
    if candidate.graph_paths:
        unit.extend(
            _card_draft(graph.edge(hop.edge_id), graph)
            for hop in candidate.graph_paths[0].hops
            if hop.edge_id not in cards
        )
    return unit


def _passage_draft(passage: Passage, candidate: Candidate, graph: AtlasGraph) -> _Draft:
    """Build the original-source evidence item for one passage."""
    if passage.entity_id is not None:
        entity = graph.entity(passage.entity_id)
        entity_ids: tuple[str, ...] = (entity.entity_id,)
        case_type = entity.case_type if entity.kind == "case_study" else None
    else:
        edge = graph.edge(passage.edge_id or "")
        entity_ids = (edge.source_id, edge.target_id)
        case_type = _case_type(edge, graph)
    return _Draft(
        item_kind="passage",
        entity_ids=entity_ids,
        passage_id=passage.passage_id,
        edge_id=passage.edge_id,
        text=f"{passage.header}\n{passage.source_text}",
        source_locator=passage.source_locator,
        case_type=case_type,
        channels=tuple(candidate.channels),
        graph_path=(candidate.graph_paths[0] if candidate.graph_paths else None),
    )


def _card_draft(edge: AtlasRelationship, graph: AtlasGraph) -> _Draft:
    """Build the deterministic structural connector card for one edge."""
    source = graph.entity(edge.source_id)
    target = graph.entity(edge.target_id)
    text = (
        f"{source.display_name} ({source.entity_id}) {edge.relation} "
        f"{target.display_name} ({target.entity_id})"
    )
    case_type = _case_type(edge, graph)
    if edge.step_id is not None:
        text = f"{text} | step {edge.step_id}"
    if case_type is not None:
        text = f"{text} | case type: {case_type}"
    return _Draft(
        item_kind="relationship",
        entity_ids=(source.entity_id, target.entity_id),
        passage_id=None,
        edge_id=edge.edge_id,
        text=text,
        source_locator=edge.source_locator,
        case_type=case_type,
        channels=(_CONNECTOR,),
        graph_path=None,
    )


def _case_type(edge: AtlasRelationship, graph: AtlasGraph) -> str | None:
    """Return the case classification of an `employs` edge, when applicable."""
    if edge.relation != "employs":
        return None
    return graph.entity(edge.source_id).case_type


def _new_drafts(unit: Sequence[_Draft], drafts: Sequence[_Draft]) -> list[_Draft]:
    """Return the drafts of one unit that the accepted packet does not hold."""
    taken = list(drafts)
    new: list[_Draft] = []
    for draft in unit:
        if _is_new(draft, taken):
            new.append(draft)
            taken.append(draft)
    return new


def _is_new(draft: _Draft, drafts: Sequence[_Draft]) -> bool:
    """Return whether a draft adds information to the tentative packet."""
    key = (draft.item_kind, draft.passage_id, draft.edge_id)
    return all(
        (other.item_kind, other.passage_id, other.edge_id) != key for other in drafts
    )


def _cover(covered: dict[str, bool], passage: Passage, graph: AtlasGraph) -> None:
    """Record which evidence categories the packet now contains."""
    if passage.entity_id is not None:
        kind = graph.entity(passage.entity_id).kind
        if kind in covered:
            covered[kind] = True
    elif passage.edge_id is not None:
        relation = graph.edge(passage.edge_id).relation
        if relation == "mitigates":
            covered["mitigation"] = True


def _connecting_cards(
    unit: Sequence[_Draft], drafts: Sequence[_Draft], graph: AtlasGraph
) -> list[_Draft]:
    """Return cards that connect the tentative selection's owners.

    Connector closure is evaluated while the new evidence unit is considered,
    so a unit never enters the packet without the cards that explain it.

    Args:
        unit (Sequence[_Draft]): Drafts of the candidate under consideration.
        drafts (Sequence[_Draft]): Drafts already accepted into the packet.
        graph (AtlasGraph): Generation graph.

    Returns:
        (list[_Draft]) Deterministic cards not yet present in the packet.
    """
    if not drafts:
        return []
    named = {entity for draft in (*drafts, *unit) for entity in draft.entity_ids}
    held = {
        draft.edge_id for draft in (*drafts, *unit) if draft.item_kind == "relationship"
    }
    sources = sorted(
        entity
        for entity in named
        if graph.entities[entity].kind in {"case_study", "mitigation"}
    )
    techniques = sorted(
        entity for entity in named if graph.entities[entity].kind == "technique"
    )
    return [
        _card_draft(edge, graph)
        for source_id in sources
        for technique_id in techniques
        for edge in _connection_edges(graph, source_id, technique_id)
        if edge.edge_id not in held
    ]


def _connection_edges(
    graph: AtlasGraph, source_id: str, technique_id: str
) -> tuple[AtlasRelationship, ...]:
    """Return the direct edge or complete parent path between two owners."""
    kind = graph.entities[source_id].kind
    relation = "employs" if kind == "case_study" else "mitigates"
    direct = graph.edges_between(source_id, technique_id, relation)
    if direct:
        return direct
    return tuple(
        edge
        for mitigates, specializes in _parent_edges(graph, source_id, technique_id)
        for edge in (mitigates, specializes)
    )


def _parent_edges(
    graph: AtlasGraph, source_id: str, technique_id: str
) -> tuple[tuple[AtlasRelationship, AtlasRelationship], ...]:
    """Return mitigation-to-parent and subtechnique-to-parent edge pairs."""
    pairs: list[tuple[AtlasRelationship, AtlasRelationship]] = []
    for mitigates in graph.edges.values():
        if mitigates.relation != "mitigates" or mitigates.source_id != source_id:
            continue
        pairs.extend(
            (mitigates, specializes)
            for specializes in graph.edges.values()
            if specializes.relation == "specializes"
            and specializes.source_id == technique_id
            and specializes.target_id == mitigates.target_id
        )
    return tuple(pairs)


def _items(drafts: Sequence[_Draft], collection_version: str) -> list[EvidenceItem]:
    """Assign review-local identifiers and validate the evidence items."""
    return [
        EvidenceItem(
            evidence_id=f"E{index:03d}",
            item_kind=draft.item_kind,
            entity_ids=list(draft.entity_ids),
            passage_id=draft.passage_id,
            edge_id=draft.edge_id,
            text=draft.text,
            source_locator=draft.source_locator,
            collection_version=collection_version,
            case_type=draft.case_type,
            retrieval_channels=list(draft.channels),
            graph_path=draft.graph_path,
        )
        for index, draft in enumerate(drafts, start=1)
    ]


def _allowlist(graph: AtlasGraph, named: set[str]) -> dict[OwnerKind, list[str]]:
    """Group the packet's entity identifiers by supported owner kind."""
    allowlist: dict[OwnerKind, list[str]] = {}
    for entity_id in sorted(named):
        kind = graph.entities[entity_id].kind
        if kind in _OWNER_KINDS:
            allowlist.setdefault(cast(OwnerKind, kind), []).append(entity_id)
    return allowlist


def _associations(
    *,
    graph: AtlasGraph,
    cards: Mapping[str, str],
    entity_items: Mapping[str, Sequence[str]],
    named: set[str],
) -> list[AllowedAssociation]:
    """Build the source relationships fully supported by this packet."""
    links = [
        link
        for source_id, technique_id in _named_pairs(graph, named)
        for link in _links(graph, source_id, technique_id)
    ]
    return [
        _association(link, cards, entity_items)
        for link in links
        if {edge.edge_id for edge in link.edges} <= set(cards)
    ]


def _named_pairs(graph: AtlasGraph, named: set[str]) -> list[tuple[str, str]]:
    """Return packet source-technique pairs in stable order."""
    sources = sorted(
        entity
        for entity in named
        if graph.entities[entity].kind in {"case_study", "mitigation"}
    )
    techniques = sorted(
        entity for entity in named if graph.entities[entity].kind == "technique"
    )
    return [(source, technique) for source in sources for technique in techniques]


def _links(graph: AtlasGraph, source_id: str, technique_id: str) -> list[_Link]:
    """Return the direct and parent-linked associations of one named pair."""
    kind = graph.entities[source_id].kind
    source_kind: SourceKind = "case_study" if kind == "case_study" else "mitigation"
    relation = "employs" if source_kind == "case_study" else "mitigates"
    direct = graph.edges_between(source_id, technique_id, relation)
    if direct:
        return [_Link(source_id, source_kind, technique_id, None, direct)]
    return [
        _Link(
            source_id,
            source_kind,
            technique_id,
            mitigates.target_id,
            (mitigates, specializes),
        )
        for mitigates, specializes in _parent_edges(graph, source_id, technique_id)
    ]


def _association(
    link: _Link,
    cards: Mapping[str, str],
    entity_items: Mapping[str, Sequence[str]],
) -> AllowedAssociation:
    """Build one validated association record with its supporting evidence."""
    evidence = [cards[edge.edge_id] for edge in link.edges]
    for entity_id in (link.source_id, link.technique_id):
        evidence.extend(entity_items.get(entity_id, ()))
    return AllowedAssociation(
        source_id=link.source_id,
        source_kind=link.source_kind,
        technique_id=link.technique_id,
        parent_technique_id=link.parent,
        edge_ids=[edge.edge_id for edge in link.edges],
        evidence_ids=list(dict.fromkeys(evidence)),
    )
