"""Parse, validate, and normalize ATLAS format-version 6 YAML sources.

Important Classes:
    ParsedCorpus: Validated records plus the identity of the source bytes.

Important Functions:
    parse_source(raw: bytes, *, origin: str, acquired_at: str, commit: str | None = None, license: str | None = None) -> ParsedCorpus: Parse one ATLAS v6 document.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from atlas_review.contracts import (
    AssociatedEntity,
    AtlasEntity,
    AtlasRelationship,
    JsonValue,
    SourceReference,
    SourceSnapshot,
)
from atlas_review.corpus.normalize import (
    derive_display_names,
    normalize_source_text,
    resolve_reference_markers,
)
from atlas_review.corpus.relationships import collect_edges, validate_steps
from atlas_review.corpus.source_support import (
    OBJECT_TYPES,
    OPTIONAL_SECTION_KINDS,
    SECTION_KINDS,
    EntityKind,
    ParseContext,
    RawRecord,
    as_optional_text,
    check_format_major,
    is_json_value,
    load_document,
    require_mapping,
    require_mapping_at,
    require_str,
    require_text,
    source_error,
    string_list,
    text_key,
)
from atlas_review.hashing import fingerprint, sha256_hex

PIPELINE_VERSION = "atlas-ingest-v1"

_SEARCHABLE_KINDS = frozenset[EntityKind]({"technique", "mitigation", "case_study"})
_ENTITY_FIELDS = frozenset(
    {
        "id",
        "object-type",
        "name",
        "description",
        "references",
        "created-date",
        "modified-date",
        "uuid",
        "platforms",
        "maturity",
        "lifecycle-phases",
        "categories",
        "type",
        "actor",
        "target",
        "date",
        "date-granularity",
        "case-study",
        "attack-reference",
        "reporter",
        "version",
        "software-type",
        "software-categories",
        "aliases",
        "entity-type",
        "associated-entities",
        "start-date",
        "end-date",
        "start-date-granularity",
        "end-date-granularity",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedCorpus:
    """Validated ATLAS records plus the identity of the bytes they came from.

    Instance Attributes:
        snapshot (SourceSnapshot): Exact source identity for the generation.
        generation_id (str): Deterministic identifier shared by all records.
        entities (tuple[AtlasEntity, ...]): Canonical entity records.
        edges (tuple[AtlasRelationship, ...]): Canonical relationship records.
        warnings (tuple[str, ...]): Sanitized non-fatal ingestion observations.

    Example Usage:
        ```python
            >>> corpus = parse_source(raw, origin="fixture", acquired_at=now)
            >>> corpus.snapshot.collection_version
            '2026.09'
        ```
    """

    snapshot: SourceSnapshot
    generation_id: str
    entities: tuple[AtlasEntity, ...]
    edges: tuple[AtlasRelationship, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def parse_source(
    raw: bytes,
    *,
    origin: str,
    acquired_at: str,
    commit: str | None = None,
    license: str | None = None,
) -> ParsedCorpus:
    """Parse one ATLAS v6 YAML document into canonical records.

    Args:
        raw (bytes): Exact source bytes, hashed for the snapshot.
        origin (str): Operator-visible provenance such as a path or URL.
        acquired_at (str): ISO 8601 acquisition timestamp with an offset.
        commit (str | None): Optional upstream revision identifier.
        license (str | None): Optional source license identifier.

    Returns:
        (ParsedCorpus) Validated entities, edges, snapshot, and warnings.

    Raises:
        AtlasReviewError: The document is unreadable, unsupported, internally
            inconsistent, or contains no searchable records.

    Example Usage:
        ```python
            >>> corpus = parse_source(
            ...     raw, origin=str(path), acquired_at=timestamp()
            ... )
        ```
    """
    document = load_document(raw)
    warnings: list[str] = []
    format_version = require_text(document, "format-version", "/format-version")
    check_format_major(format_version)
    collection = require_mapping(document, "collection", "/collection")
    records = _collect_records(document, warnings)
    context = ParseContext(
        generation_id=_generation_id(raw),
        entities={record.record_id: record for record in records},
        warnings=warnings,
    )
    edges = collect_edges(document, context)
    validate_steps(edges)
    if not any(record.kind in _SEARCHABLE_KINDS for record in records):
        raise source_error(
            "empty_corpus", "source contains no searchable ATLAS records"
        )
    snapshot = SourceSnapshot(
        origin=origin,
        format_version=format_version,
        collection_version=require_text(collection, "version", "/collection/version"),
        acquired_at=acquired_at,
        source_sha256=sha256_hex(raw),
        commit=commit,
        license=license,
    )
    return ParsedCorpus(
        snapshot=snapshot,
        generation_id=context.generation_id,
        entities=_build_entities(records, context.generation_id, edges, warnings),
        edges=edges,
        warnings=tuple(warnings),
    )


def _generation_id(raw: bytes) -> str:
    """Derive a stable generation identifier from the source bytes."""
    seed: JsonValue = {
        "source_sha256": sha256_hex(raw),
        "pipeline": PIPELINE_VERSION,
    }
    return str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint(seed)))


def _collect_records(
    document: Mapping[str, Any], warnings: list[str]
) -> list[RawRecord]:
    """Collect nested-source and section records in source order."""
    records = [
        _record(
            require_mapping(document, "collection", "/collection"),
            "collection",
            "/collection",
            warnings,
        ),
        _record(
            require_mapping(document, "matrix", "/matrix"),
            "matrix",
            "/matrix",
            warnings,
        ),
    ]
    for section, kind in SECTION_KINDS.items():
        records.extend(_section_records(document, section, kind, warnings))
    for section, kind in OPTIONAL_SECTION_KINDS.items():
        if section in document:
            records.extend(_section_records(document, section, kind, warnings))
    seen: set[str] = set()
    for record in records:
        if record.record_id in seen:
            raise source_error(
                "duplicate_record_id",
                f"record identifier {record.record_id!r} appears more than once",
                (record.locator,),
            )
        seen.add(record.record_id)
    return records


def _section_records(
    document: Mapping[str, Any], section: str, kind: EntityKind, warnings: list[str]
) -> list[RawRecord]:
    """Build records for one keyed section such as ``techniques``."""
    section_map = require_mapping(document, section, f"/{section}")
    records: list[RawRecord] = []
    for key in section_map:
        path = f"/{section}/{key}"
        data = require_mapping_at(section_map[key], path)
        record = _record(data, kind, path, warnings)
        if record.record_id != key:
            raise source_error(
                "record_key_mismatch",
                f"record key {key!r} does not match its id {record.record_id!r}",
                (path,),
            )
        records.append(record)
    return records


def _record(
    data: Mapping[str, Any], kind: EntityKind, locator: str, warnings: list[str]
) -> RawRecord:
    """Validate one source record and normalize its text."""
    record_id = require_text(data, "id", f"{locator}/id")
    expected_type = OBJECT_TYPES[kind]
    if data.get("object-type") != expected_type:
        raise source_error(
            "unsupported_object_type",
            f"record {record_id!r} must declare object-type {expected_type!r}",
            (f"{locator}/object-type",),
        )
    name = require_text(data, "name", f"{locator}/name")
    description = normalize_source_text(require_str(data, "description", locator))
    if not description:
        raise source_error(
            "empty_record_description",
            f"record {record_id!r} has an empty description",
            (f"{locator}/description",),
        )
    _warn_absent_dates(data, locator, warnings)
    return RawRecord(
        record_id=record_id,
        kind=kind,
        locator=locator,
        name=name,
        data=data,
        description=description,
        references=_references(data.get("references"), locator, warnings),
        source_fields=_source_fields(data, kind, locator, warnings),
    )


def _references(
    raw: Any, locator: str, warnings: list[str]
) -> tuple[SourceReference, ...]:
    """Build bibliography entries, dropping malformed or unsafe links."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        warnings.append(f"ignored malformed reference list at {locator}/references")
        return ()
    references: list[SourceReference] = []
    for index, item in enumerate(raw):
        reference = _reference(item, f"{locator}/references/{index}", warnings)
        if reference is not None:
            references.append(reference)
    return tuple(references)


def _reference(item: Any, path: str, warnings: list[str]) -> SourceReference | None:
    """Build one bibliography entry, warning instead of failing on bad data."""
    if not isinstance(item, Mapping):
        warnings.append(f"ignored malformed source reference at {path}")
        return None
    ref_id = item.get("id")
    title = item.get("title")
    if not isinstance(ref_id, str) or not ref_id.strip():
        warnings.append(f"ignored source reference without an id at {path}")
        return None
    if not isinstance(title, str) or not title.strip():
        warnings.append(
            f"ignored source reference {ref_id!r} without a title at {path}"
        )
        return None
    url = item.get("url")
    link: str | None = None
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        link = url
    elif url is not None:
        warnings.append(
            f"source reference {ref_id!r} has a URL that is not an ordinary "
            f"HTTP(S) link; stored without a link ({path})"
        )
    return SourceReference(ref_id=ref_id.strip(), title=title.strip(), url=link)


def _source_fields(
    data: Mapping[str, Any], kind: EntityKind, locator: str, warnings: list[str]
) -> dict[str, JsonValue]:
    """Retain additive source fields that survive JSON encoding."""
    retained: dict[str, JsonValue] = {}
    for raw_key, value in data.items():
        key = text_key(raw_key)
        if key is None or key in _ENTITY_FIELDS:
            continue
        if is_json_value(value):
            retained[key] = value
            warnings.append(
                f"retained additive source field {key!r} for {kind} at {locator}"
            )
        else:
            warnings.append(
                f"ignored non-JSON source field {key!r} for {kind} at {locator}"
            )
    return retained


def _warn_absent_dates(
    data: Mapping[str, Any], locator: str, warnings: list[str]
) -> None:
    """Record a warning for each absent optional source date."""
    warnings.extend(
        f"absent optional source date {key!r} at {locator}"
        for key in ("created-date", "modified-date")
        if data.get(key) is None
    )


def _build_entities(
    records: Sequence[RawRecord],
    generation_id: str,
    edges: Sequence[AtlasRelationship],
    warnings: list[str],
) -> tuple[AtlasEntity, ...]:
    """Create canonical entities with resolved markers and display names."""
    provisional = [_entity(record, generation_id, warnings) for record in records]
    names = derive_display_names(provisional, edges)
    return tuple(
        AtlasEntity.model_validate(
            {**entity.model_dump(), "display_name": names[entity.entity_id]}
        )
        for entity in provisional
    )


def _entity(record: RawRecord, generation_id: str, warnings: list[str]) -> AtlasEntity:
    """Create one canonical entity from a validated source record."""
    data = record.data
    description, marker_warnings = resolve_reference_markers(
        record.description, record.references
    )
    warnings.extend(marker_warnings)
    return AtlasEntity(
        generation_id=generation_id,
        entity_id=record.record_id,
        kind=record.kind,
        name=record.name,
        display_name=record.name,
        description=description,
        source_locator=f"{record.locator}/description",
        references=list(record.references),
        source_fields=record.source_fields,
        uuid=as_optional_text(data.get("uuid")),
        created_date=as_optional_text(data.get("created-date")),
        modified_date=as_optional_text(data.get("modified-date")),
        platforms=string_list(data.get("platforms")),
        maturity=as_optional_text(data.get("maturity")),
        lifecycle_phases=string_list(data.get("lifecycle-phases")),
        categories=string_list(data.get("categories")),
        case_type=as_optional_text(data.get("type")),
        case_actor=as_optional_text(data.get("actor")),
        case_target=as_optional_text(data.get("target")),
        case_date=as_optional_text(data.get("date")),
        case_date_granularity=as_optional_text(data.get("date-granularity")),
        case_start_date=as_optional_text(data.get("start-date")),
        case_start_date_granularity=as_optional_text(
            data.get("start-date-granularity")
        ),
        case_end_date=as_optional_text(data.get("end-date")),
        case_end_date_granularity=as_optional_text(data.get("end-date-granularity")),
        attack_reference=_attack_reference(data.get("attack-reference")),
        reporter=as_optional_text(data.get("reporter")),
        software_type=as_optional_text(data.get("software-type")),
        software_categories=string_list(data.get("software-categories")),
        aliases=string_list(data.get("aliases")),
        entity_type=as_optional_text(data.get("entity-type")),
        associated_entities=_associated_entities(
            data.get("associated-entities"), record.locator, warnings
        ),
    )


def _attack_reference(raw: Any) -> str | None:
    """Reduce an ATT&CK reference mapping to its identifier."""
    if isinstance(raw, Mapping):
        return as_optional_text(raw.get("id"))
    return None


def _associated_entities(
    raw: Any, locator: str, warnings: list[str]
) -> list[AssociatedEntity]:
    """Read an entity's cross-references, warning instead of failing on bad data."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append(f"ignored malformed associated entities at {locator}")
        return []
    entities: list[AssociatedEntity] = []
    for index, item in enumerate(raw):
        path = f"{locator}/associated-entities/{index}"
        if not isinstance(item, Mapping):
            warnings.append(f"ignored malformed associated entity at {path}")
            continue
        entity_id = item.get("id")
        name = item.get("name")
        if not isinstance(entity_id, str) or not isinstance(name, str):
            warnings.append(
                f"ignored associated entity without an identifier and name at {path}"
            )
            continue
        entities.append(AssociatedEntity(entity_id=entity_id, name=name))
    return entities
