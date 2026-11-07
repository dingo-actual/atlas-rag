"""Shared ATLAS source-schema constants and YAML value accessors.

Important Functions:
    source_error(code: str, message: str, details: Iterable[str] = ()) -> AtlasReviewError: Build a schema error with the standard recovery hint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import yaml
from yaml.nodes import MappingNode, ScalarNode

from atlas_review.contracts import JsonValue, SourceReference
from atlas_review.errors import STAGE_SOURCE_SCHEMA, AtlasReviewError

SUPPORTED_FORMAT = "6.x"
SUPPORTED_FORMAT_MAJOR = 6
DOWNLOAD_URL = "https://github.com/mitre-atlas/atlas-data"

EntityKind = Literal[
    "collection", "matrix", "tactic", "technique", "mitigation", "case_study"
]
RelationName = Literal["employs", "mitigates", "specializes", "achieves", "sequences"]

SECTION_KINDS: dict[str, EntityKind] = {
    "tactics": "tactic",
    "techniques": "technique",
    "mitigations": "mitigation",
    "case-studies": "case_study",
}
OBJECT_TYPES: dict[EntityKind, str] = {
    "collection": "collection",
    "matrix": "matrix",
    "tactic": "tactic",
    "technique": "technique",
    "mitigation": "mitigation",
    "case_study": "case-study",
}
RELATION_ENDPOINTS: dict[RelationName, tuple[EntityKind, EntityKind]] = {
    "employs": ("case_study", "technique"),
    "mitigates": ("mitigation", "technique"),
    "specializes": ("technique", "technique"),
    "achieves": ("technique", "tactic"),
    "sequences": ("matrix", "tactic"),
}

MAX_DECODED_NODES = 1_000_000
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"


class SourceLoader(yaml.SafeLoader):
    """Safe loader that keeps source dates textual and rejects duplicate keys.

    Implicit timestamp resolution is redirected to the source's own scalar text,
    so ``created-date: 2024-01-01`` stays a string rather than becoming a
    Python date object.
    """

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        """Reject an authored duplicate key instead of silently overwriting it."""
        seen: set[tuple[str, str]] = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, ScalarNode):
                continue
            marker = (key_node.tag, key_node.value)
            if marker in seen:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate mapping key {key_node.value!r}",
                    key_node.start_mark,
                )
            seen.add(marker)
        return super().construct_mapping(node, deep=deep)


SourceLoader.add_constructor(
    _TIMESTAMP_TAG, lambda _loader, node: cast("str", node.value)
)


@dataclass(frozen=True, slots=True)
class RawRecord:
    """One source record with normalized text before cross-record checks.

    Instance Attributes:
        record_id (str): Source ``id`` value used as the canonical identifier.
        kind (EntityKind): Record kind derived from its source location.
        locator (str): Source locator prefix such as ``/techniques/AML.T0051``.
        name (str): Source display name.
        data (Mapping[str, Any]): Raw record mapping for optional attributes.
        description (str): Normalized description text with markers unresolved.
        references (tuple[SourceReference, ...]): Safe bibliography entries.
        source_fields (dict[str, JsonValue]): Retained additive source fields.
    """

    record_id: str
    kind: EntityKind
    locator: str
    name: str
    data: Mapping[str, Any]
    description: str
    references: tuple[SourceReference, ...]
    source_fields: dict[str, JsonValue]


@dataclass(slots=True)
class ParseContext:
    """Mutable parse state shared by record and edge collection.

    Instance Attributes:
        generation_id (str): Deterministic generation identifier.
        entities (dict[str, RawRecord]): Records keyed by source identifier.
        warnings (list[str]): Accumulated sanitized observations.
    """

    generation_id: str
    entities: dict[str, RawRecord]
    warnings: list[str]


def source_error(
    code: str, message: str, details: Iterable[str] = (), recovery: str = ""
) -> AtlasReviewError:
    """Build a source-schema error with the standard recovery hint.

    Args:
        code (str): Stable machine-readable error code.
        message (str): Short sanitized message.
        details (tuple[str, ...]): Optional locator or expectation lines.
        recovery (str): Optional override for the default recovery hint.

    Returns:
        (AtlasReviewError) Error bound to the source-schema stage.
    """
    return AtlasReviewError(
        code=code,
        stage=STAGE_SOURCE_SCHEMA,
        message=message,
        details=tuple(details),
        recovery=recovery
        or (
            f"Check the source against ATLAS format {SUPPORTED_FORMAT} and "
            f"re-download it from {DOWNLOAD_URL}."
        ),
    )


def load_document(raw: bytes) -> Mapping[str, Any]:
    """Decode the YAML document and require a bounded, acyclic mapping.

    Args:
        raw (bytes): Source bytes.

    Returns:
        (Mapping[str, Any]) Parsed top-level mapping.

    Raises:
        AtlasReviewError: The bytes are not YAML, not a mapping, contain a
            duplicate key, or decode to an oversized or cyclic structure.
    """
    try:
        document = yaml.load(raw, Loader=SourceLoader)  # noqa: S506 - safe subclass
    except yaml.constructor.ConstructorError as exc:
        raise source_error(
            "duplicate_mapping_key", exc.problem or "duplicate mapping key"
        ) from exc
    except yaml.YAMLError as exc:
        raise source_error(
            "unreadable_source", "source file is not valid YAML"
        ) from exc
    if not isinstance(document, Mapping):
        raise source_error("unreadable_source", "source file must be a YAML mapping")
    _check_decoded_shape(document)
    return document


def _check_decoded_shape(document: Mapping[str, Any]) -> None:
    """Reject cyclic aliases and structures too large to be a real collection."""
    budget = MAX_DECODED_NODES
    stack: list[tuple[object, frozenset[int]]] = [(document, frozenset())]
    while stack:
        value, ancestors = stack.pop()
        budget -= 1
        if budget < 0:
            raise source_error(
                "oversized_source_structure",
                f"source decodes to more than {MAX_DECODED_NODES} values",
            )
        children: Iterable[object] = ()
        if isinstance(value, Mapping):
            children = (*value.keys(), *value.values())
        elif isinstance(value, list):
            children = value
        else:
            continue
        if id(value) in ancestors:
            raise source_error(
                "cyclic_source_structure",
                "source contains a recursive alias",
            )
        nested = ancestors | {id(value)}
        stack.extend((child, nested) for child in children)


def check_format_major(format_version: str) -> None:
    """Reject a document whose major format version is not supported.

    Args:
        format_version (str): Declared ``format-version`` value.

    Raises:
        AtlasReviewError: The major version is not supported.
    """
    major_text = format_version.split(".", maxsplit=1)[0]
    if not major_text.isdigit() or int(major_text) != SUPPORTED_FORMAT_MAJOR:
        raise source_error(
            "unsupported_source_format",
            f"unsupported ATLAS format {format_version!r}",
            (f"accepted format: {SUPPORTED_FORMAT}",),
        )


def require_text(data: Mapping[str, Any], key: str, locator: str) -> str:
    """Read a required non-blank string field.

    Args:
        data (Mapping[str, Any]): Mapping that should hold the field.
        key (str): Field name.
        locator (str): Source locator reported on failure.

    Returns:
        (str) Stripped field value.

    Raises:
        AtlasReviewError: The field is missing, non-string, or blank.
    """
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise source_error(
            "invalid_source_record",
            f"required text field {key!r} is missing or blank",
            (locator,),
        )
    return value.strip()


def require_str(data: Mapping[str, Any], key: str, locator: str) -> str:
    """Read a required string field that may be blank after normalization.

    Args:
        data (Mapping[str, Any]): Mapping that should hold the field.
        key (str): Field name.
        locator (str): Source locator reported on failure.

    Returns:
        (str) Unmodified field value.

    Raises:
        AtlasReviewError: The field is missing or not a string.
    """
    value = data.get(key)
    if not isinstance(value, str):
        raise source_error(
            "invalid_source_record",
            f"required text field {key!r} is missing",
            (locator,),
        )
    return value


def require_mapping(
    data: Mapping[str, Any], key: str, locator: str
) -> Mapping[str, Any]:
    """Read a required mapping field.

    Args:
        data (Mapping[str, Any]): Mapping that should hold the field.
        key (str): Field name.
        locator (str): Source locator reported on failure.

    Returns:
        (Mapping[str, Any]) Field value.

    Raises:
        AtlasReviewError: The field is missing or not a mapping.
    """
    return require_mapping_at(data.get(key), locator)


def require_mapping_at(value: Any, locator: str) -> Mapping[str, Any]:
    """Require a mapping value at an already-addressed locator.

    Args:
        value (Any): Candidate value.
        locator (str): Source locator reported on failure.

    Returns:
        (Mapping[str, Any]) The value, narrowed to a mapping.

    Raises:
        AtlasReviewError: The value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise source_error(
            "invalid_source_record", f"expected a mapping at {locator}", (locator,)
        )
    return value


def is_json_value(value: object) -> bool:
    """Report whether a loader value can be stored in a JSON column.

    Args:
        value (object): Candidate value from the YAML loader.

    Returns:
        (bool) ``True`` when the value is JSON-compatible.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and is_json_value(item) for key, item in value.items()
        )
    return False


def text_key(key: object) -> str | None:
    """Return a mapping key when it is text; YAML can load non-string keys.

    Args:
        key (object): Mapping key from the YAML loader.

    Returns:
        (str | None) The key when it is a string.
    """
    return key if isinstance(key, str) else None


def as_optional_text(value: Any) -> str | None:
    """Return a string field, or ``None`` for absent or non-string values.

    Args:
        value (Any): Candidate value.

    Returns:
        (str | None) The value when it is a string.
    """
    return value if isinstance(value, str) else None


def string_list(raw: Any) -> list[str]:
    """Read a list of strings, ignoring blank and non-string entries.

    Args:
        raw (Any): Candidate list value.

    Returns:
        (list[str]) Stripped string entries.
    """
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
