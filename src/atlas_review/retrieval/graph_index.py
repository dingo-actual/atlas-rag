"""rustworkx reconstruction of one generation's ATLAS entity graph.

The graph is derived from stored entity and edge rows, keeps parallel edges,
and carries the original relationship orientation. rustworkx node and edge
indices are runtime details; ATLAS identifiers stay the stable identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import rustworkx as rx

from atlas_review.contracts import AtlasEntity, AtlasRelationship
from atlas_review.errors import STAGE_STORAGE, AtlasReviewError


@dataclass(frozen=True, slots=True)
class AtlasGraph:
    """One generation's directed multigraph and its identifier lookups.

    Instance Attributes:
        graph (rx.PyDiGraph): Directed multigraph of entities and edges.
        node_by_id (dict[str, int]): Entity identifier to node index.
        edges_by_id (dict[str, int]): Edge identifier to edge index.
        entities (dict[str, AtlasEntity]): Stored entities by identifier.
        edges (dict[str, AtlasRelationship]): Stored edges by identifier.
    """

    graph: rx.PyDiGraph
    node_by_id: dict[str, int]
    edges_by_id: dict[str, int]
    entities: dict[str, AtlasEntity]
    edges: dict[str, AtlasRelationship]

    def entity(self, entity_id: str) -> AtlasEntity:
        """Return one stored entity.

        Args:
            entity_id (str): ATLAS identifier.

        Returns:
            (AtlasEntity) Stored entity record.

        Raises:
            AtlasReviewError: The identifier is not part of the generation.
        """
        entity = self.entities.get(entity_id)
        if entity is None:
            raise _graph_error(
                "unknown_entity",
                f"entity {entity_id} is not part of this generation",
            )
        return entity

    def edge(self, edge_id: str) -> AtlasRelationship:
        """Return one stored relationship edge.

        Args:
            edge_id (str): Stored edge identifier.

        Returns:
            (AtlasRelationship) Stored edge record.

        Raises:
            AtlasReviewError: The edge identifier is not part of the generation.
        """
        edge = self.edges.get(edge_id)
        if edge is None:
            raise _graph_error(
                "unknown_edge", f"edge {edge_id} is not part of this generation"
            )
        return edge

    def edge_at(self, index: int) -> AtlasRelationship:
        """Return the stored relationship edge at one rustworkx index.

        Args:
            index (int): Edge index inside the graph.

        Returns:
            (AtlasRelationship) Stored edge record.
        """
        return cast(AtlasRelationship, self.graph.get_edge_data_by_index(index))

    def edges_between(
        self, source_id: str, target_id: str, relation: str
    ) -> tuple[AtlasRelationship, ...]:
        """Return every stored edge between two identifiers of one relation.

        Args:
            source_id (str): Original source entity identifier.
            target_id (str): Original target entity identifier.
            relation (str): Relationship type to keep.

        Returns:
            (tuple[AtlasRelationship, ...]) Parallel edges in stable order.
        """
        source = self.node_by_id.get(source_id)
        target = self.node_by_id.get(target_id)
        if source is None or target is None:
            return ()
        indices = sorted(self.graph.edge_indices_from_endpoints(source, target))
        return tuple(
            self.edge_at(index)
            for index in indices
            if self.edge_at(index).relation == relation
        )


def build_graph(
    entities: Sequence[AtlasEntity], edges: Sequence[AtlasRelationship]
) -> AtlasGraph:
    """Build the directed multigraph for one generation.

    Args:
        entities (Sequence[AtlasEntity]): Stored entities.
        edges (Sequence[AtlasRelationship]): Stored relationship edges.

    Returns:
        (AtlasGraph) Graph with identifier lookups.

    Raises:
        AtlasReviewError: An edge names an entity outside the generation.
    """
    ordered_entities = sorted(entities, key=lambda entity: entity.entity_id)
    ordered_edges = sorted(edges, key=lambda edge: edge.edge_id)
    graph: rx.PyDiGraph = rx.PyDiGraph(multigraph=True)
    node_by_id: dict[str, int] = {}
    for entity in ordered_entities:
        node_by_id[entity.entity_id] = graph.add_node(entity)
    edges_by_id: dict[str, int] = {}
    for edge in ordered_edges:
        source = node_by_id.get(edge.source_id)
        target = node_by_id.get(edge.target_id)
        if source is None or target is None:
            raise _graph_error(
                "dangling_edge",
                f"edge {edge.edge_id} names an entity outside this generation",
            )
        edges_by_id[edge.edge_id] = graph.add_edge(source, target, edge)
    return AtlasGraph(
        graph=graph,
        node_by_id=node_by_id,
        edges_by_id=edges_by_id,
        entities={entity.entity_id: entity for entity in ordered_entities},
        edges={edge.edge_id: edge for edge in ordered_edges},
    )


def _graph_error(code: str, message: str) -> AtlasReviewError:
    """Build a graph-reconstruction error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_STORAGE,
        message=message,
        recovery="Re-ingest the source so every relationship endpoint resolves.",
    )
