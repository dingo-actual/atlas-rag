"""Relationship-edge validation for ATLAS source documents.

Important Functions:
    collect_edges(document: Mapping[str, Any], context: ParseContext) -> tuple[AtlasRelationship, ...]: Build every relationship edge.
    validate_steps(edges: Sequence[AtlasRelationship]) -> None: Require case-scoped step identifiers to resolve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from atlas_review.contracts import AtlasRelationship, JsonValue
from atlas_review.corpus.normalize import (
    normalize_optional,
    resolve_reference_markers,
)
from atlas_review.corpus.source_support import (
    RELATION_ENDPOINTS,
    ParseContext,
    RawRecord,
    RelationName,
    as_optional_text,
    is_json_value,
    require_mapping,
    require_mapping_at,
    require_text,
    source_error,
    text_key,
)
from atlas_review.hashing import fingerprint

_RELATIONSHIP_FIELDS = frozenset(
    {
        "source",
        "target",
        "relationship-type",
        "description",
        "step-id",
        "tactic",
        "leads-to",
        "position",
    }
)


def collect_edges(
    document: Mapping[str, Any], context: ParseContext
) -> tuple[AtlasRelationship, ...]:
    """Build every relationship edge, preserving source order and orientation.

    Args:
        document (Mapping[str, Any]): Parsed source document.
        context (ParseContext): Records and warnings gathered so far.

    Returns:
        (tuple[AtlasRelationship, ...]) Edges in source order.

    Raises:
        AtlasReviewError: An edge names an unknown record, an unsupported
            relationship type, or invalid field values.
    """
    relationships = require_mapping(document, "relationships", "/relationships")
    edges: list[AtlasRelationship] = []
    for source_key in relationships:
        block = require_mapping_at(
            relationships[source_key], f"/relationships/{source_key}"
        )
        owner = context.entities.get(source_key)
        if owner is None:
            raise source_error(
                "unknown_relationship_source",
                f"relationships reference unknown record {source_key!r}",
                (f"/relationships/{source_key}",),
            )
        for relation in block:
            edges.extend(
                _relation_edges(
                    block[relation], source_key, str(relation), owner, context
                )
            )
    return tuple(edges)


def validate_steps(edges: Sequence[AtlasRelationship]) -> None:
    """Require case-scoped step identifiers and ``leads-to`` links to resolve.

    Args:
        edges (Sequence[AtlasRelationship]): Collected edges.

    Raises:
        AtlasReviewError: A case repeats a step identifier or links to a step
            that the same case study does not define.
    """
    steps: dict[str, set[str]] = {}
    for edge in edges:
        if edge.relation != "employs" or edge.step_id is None:
            continue
        case_steps = steps.setdefault(edge.source_id, set())
        if edge.step_id in case_steps:
            raise source_error(
                "duplicate_step_id",
                f"case {edge.source_id!r} repeats step identifier {edge.step_id!r}",
                (edge.source_locator,),
            )
        case_steps.add(edge.step_id)
    for edge in edges:
        for target_step in edge.leads_to:
            if target_step not in steps.get(edge.source_id, set()):
                raise source_error(
                    "unresolved_step_reference",
                    f"case {edge.source_id!r} links to unknown step {target_step!r}",
                    (edge.source_locator,),
                )


def _relation_edges(
    raw: Any,
    source_key: str,
    relation: str,
    owner: RawRecord,
    context: ParseContext,
) -> list[AtlasRelationship]:
    """Build the edges of one source record and one relationship type."""
    if relation not in RELATION_ENDPOINTS:
        raise source_error(
            "unsupported_relationship_type",
            f"unsupported relationship type {relation!r}",
            (
                f"/relationships/{source_key}/{relation}",
                f"accepted types: {', '.join(sorted(RELATION_ENDPOINTS))}",
            ),
        )
    if not isinstance(raw, list):
        raise source_error(
            "invalid_source_record",
            f"relationship type {relation!r} must hold a list of edges",
            (f"/relationships/{source_key}/{relation}",),
        )
    return [
        _edge(
            item,
            source_key,
            relation,
            index=index,
            owner=owner,
            context=context,
        )
        for index, item in enumerate(raw)
    ]


def _edge(
    item: Any,
    source_key: str,
    relation: RelationName,
    *,
    index: int,
    owner: RawRecord,
    context: ParseContext,
) -> AtlasRelationship:
    """Validate one edge record and normalize its description."""
    locator = f"/relationships/{source_key}/{relation}/{index}"
    data = require_mapping_at(item, locator)
    if data.get("source", source_key) != source_key:
        raise source_error(
            "relationship_source_mismatch",
            f"edge at {locator} declares a different source",
            (locator,),
        )
    if data.get("relationship-type", relation) != relation:
        raise source_error(
            "relationship_type_mismatch",
            f"edge at {locator} declares a different relationship type",
            (locator,),
        )
    target_id = require_text(data, "target", f"{locator}/target")
    target = context.entities.get(target_id)
    if target is None:
        raise source_error(
            "unknown_relationship_target",
            f"edge at {locator} targets unknown record {target_id!r}",
            (locator,),
        )
    _check_endpoint_kinds(relation, owner, target, locator)
    description = normalize_optional(as_optional_text(data.get("description")))
    if description is not None:
        description, marker_warnings = resolve_reference_markers(
            description, owner.references
        )
        context.warnings.extend(marker_warnings)
    step_id = normalize_optional(as_optional_text(data.get("step-id")))
    tactic_id = normalize_optional(as_optional_text(data.get("tactic")))
    if tactic_id is not None:
        _warn_unknown_tactic(tactic_id, owner, context)
    return AtlasRelationship(
        generation_id=context.generation_id,
        edge_id=fingerprint(
            {
                "relation": relation,
                "source_id": source_key,
                "target_id": target_id,
                "step_id": step_id,
                "tactic_id": tactic_id,
                "source_index": index,
                "source_locator": locator,
            }
        ),
        relation=relation,
        source_id=source_key,
        target_id=target_id,
        description=description,
        step_id=step_id,
        tactic_id=tactic_id,
        position=_position(relation, data.get("position"), index, locator),
        leads_to=_leads_to(data.get("leads-to"), locator),
        source_locator=locator,
        source_fields=_relationship_source_fields(data, locator, context),
    )


def _check_endpoint_kinds(
    relation: RelationName, source: RawRecord, target: RawRecord, locator: str
) -> None:
    """Reject edges whose endpoints have the wrong record kinds."""
    expected_source, expected_target = RELATION_ENDPOINTS[relation]
    if source.kind != expected_source or target.kind != expected_target:
        raise source_error(
            "relationship_endpoint_kind",
            f"relationship {relation!r} must connect {expected_source} to "
            f"{expected_target}",
            (locator,),
        )


def _warn_unknown_tactic(
    tactic_id: str, owner: RawRecord, context: ParseContext
) -> None:
    """Warn when an ``employs`` edge names a tactic that is not in the source."""
    record = context.entities.get(tactic_id)
    if record is None or record.kind != "tactic":
        context.warnings.append(
            f"edge at {owner.locator} names unknown tactic {tactic_id!r}"
        )


def _leads_to(raw: Any, locator: str) -> list[str]:
    """Validate a ``leads-to`` list of case-scoped step identifiers."""
    if raw is None:
        return []
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise source_error(
            "invalid_source_record",
            f"leads-to at {locator} must list step identifiers",
            (locator,),
        )
    return [item.strip() for item in raw]


def _position(relation: RelationName, raw: Any, index: int, locator: str) -> int | None:
    """Read the source position for matrix order or case-step order."""
    if relation == "employs":
        return index
    if relation != "sequences":
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise source_error(
            "invalid_source_record",
            f"position at {locator} must be a non-negative integer",
            (locator,),
        )
    return raw


def _relationship_source_fields(
    data: Mapping[str, Any], locator: str, context: ParseContext
) -> dict[str, JsonValue]:
    """Retain additive relationship fields that survive JSON encoding."""
    retained: dict[str, JsonValue] = {}
    for raw_key, value in data.items():
        key = text_key(raw_key)
        if key is None or key in _RELATIONSHIP_FIELDS:
            continue
        if is_json_value(value):
            retained[key] = value
            context.warnings.append(
                f"retained additive source field {key!r} for relationship at {locator}"
            )
        else:
            context.warnings.append(
                f"ignored non-JSON source field {key!r} for relationship at {locator}"
            )
    return retained
