"""Conservative normalization of ATLAS source text for display and retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser

from atlas_review.contracts import AtlasEntity, AtlasRelationship, SourceReference

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_SPACE = re.compile(r"[ \t\f\v]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_MARKER = re.compile(r"\[\[([^\[\]]{1,200})\]\]")

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "pre",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "table",
        "section",
        "article",
        "hr",
        "dd",
        "dt",
        "figure",
    }
)
_HIDDEN_TAGS = frozenset({"script", "style"})


class _VisibleTextParser(HTMLParser):
    """Keep visible text and line boundaries; never interpret or fetch markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN_TAGS:
            self._hidden_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN_TAGS:
            self._hidden_depth = max(0, self._hidden_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def normalize_source_text(raw: str) -> str:
    """Decode entities and markup, then standardize whitespace."""
    parser = _VisibleTextParser()
    parser.feed(raw)
    parser.close()
    text = parser.text().replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARACTERS.sub("", text)
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def normalize_optional(raw: str | None) -> str | None:
    """Return normalized text, or ``None`` when the source field is blank."""
    if raw is None:
        return None
    normalized = normalize_source_text(raw)
    return normalized or None


def resolve_reference_markers(
    text: str, references: Sequence[SourceReference]
) -> tuple[str, list[str]]:
    """Resolve ``[[ref-id]]`` markers against the owning record's references."""
    by_id = {reference.ref_id: reference for reference in references}
    warnings: list[str] = []

    def replace(match: re.Match[str]) -> str:
        reference = by_id.get(match.group(1))
        if reference is None:
            warnings.append(f"unresolved bibliography marker [[{match.group(1)}]]")
            return match.group(0)
        if reference.url is None:
            return f"[{reference.title}]"
        return f"[{reference.title}]({reference.url})"

    return _MARKER.sub(replace, text), warnings


def derive_display_names(
    entities: Iterable[AtlasEntity], edges: Iterable[AtlasRelationship]
) -> dict[str, str]:
    """Derive ``Parent / Child`` display names from ``specializes`` edges."""
    parents: dict[str, list[str]] = {}
    for edge in edges:
        if edge.relation == "specializes":
            parents.setdefault(edge.source_id, []).append(edge.target_id)
    names = {entity.entity_id: entity.name for entity in entities}
    resolved: dict[str, str] = {}

    def resolve(entity_id: str, visiting: frozenset[str]) -> str:
        if entity_id in resolved:
            return resolved[entity_id]
        parent_ids = sorted(parents.get(entity_id, ()))
        if not parent_ids or entity_id in visiting:
            return names[entity_id]
        parent = resolve(parent_ids[0], visiting | {entity_id})
        resolved[entity_id] = f"{parent} / {names[entity_id]}"
        return resolved[entity_id]

    for entity in entities:
        resolved.setdefault(entity.entity_id, resolve(entity.entity_id, frozenset()))
    return resolved
