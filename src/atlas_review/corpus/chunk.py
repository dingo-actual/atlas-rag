"""Token-aware source chunking with space reserved for headers and context.

Chunk windows are measured with the retrieval tokenizers, keep absolute
normalized-text offsets, and always advance so a build cannot loop forever.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from atlas_review.config import ChunkingConfig
from atlas_review.contracts import AtlasEntity, AtlasRelationship, OwnerKind
from atlas_review.retrieval.tokenizers import TokenCounter

LAYOUT_VERSION = "atlas-passage-layout-v1"

_SEARCHABLE_KINDS = frozenset({"technique", "mitigation", "case_study"})
_DESCRIPTION_RELATIONS = frozenset({"employs", "mitigates"})
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_SENTENCE_BREAK = re.compile(r"[^.!?\n]+[.!?]*")

_KIND_LABELS = {
    "technique": "Technique",
    "mitigation": "Mitigation",
    "case_study": "Case",
}


@dataclass(frozen=True, slots=True)
class TextWindow:
    """One chunk of normalized source text with absolute offsets.

    Instance Attributes:
        start (int): Offset of the first character in the normalized text.
        end (int): Offset one past the last character.
        text (str): Exact ``text[start:end]`` slice.
    """

    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """One searchable source excerpt with its deterministic header.

    Instance Attributes:
        owner_kind (str): Kind of the owner document that contextualizes it.
        owner_id (str): ATLAS ID of the owner document.
        entity_id (str | None): Owning entity, when the excerpt is a description.
        edge_id (str | None): Owning relationship, when the excerpt is a step.
        section (str): Source section such as ``description`` or ``employs``.
        ordinal (int): Zero-based chunk position within the section.
        source_locator (str): Absolute locator of the excerpt.
        header (str): Deterministic retrieval header inside the token budget.
        text (str): Normalized source excerpt.
        start (int): Offset of the excerpt in the owner's description text.
        end (int): Offset one past the excerpt.
    """

    owner_kind: OwnerKind
    owner_id: str
    entity_id: str | None
    edge_id: str | None
    section: str
    ordinal: int
    source_locator: str
    header: str
    text: str
    start: int
    end: int


def chunk_windows(
    text: str,
    counter: TokenCounter,
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[TextWindow, ...]:
    """Split normalized text into ordered windows inside the body budget.

    Args:
        text (str): Normalized source text.
        counter (TokenCounter): Tokenizer used for every measurement.
        target_tokens (int): Packing goal for one window's own text.
        max_tokens (int): Hard limit for a window, including overlap.
        overlap_tokens (int): Tokens repeated from the previous window.

    Returns:
        (tuple[TextWindow, ...]) Ordered, non-overlapping-offset windows.

    Raises:
        ValueError: The budgets are not positive or the overlap leaves no room.
    """
    if target_tokens <= 0 or max_tokens <= 0:
        raise ValueError("token budgets must be positive")
    if target_tokens > max_tokens:
        raise ValueError("target_tokens must not exceed max_tokens")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    if not text.strip():
        return ()
    atoms = _split_oversized(
        text, _paragraphs(text), counter, max_tokens - overlap_tokens
    )
    windows: list[TextWindow] = []
    cursor = 0
    while cursor < len(atoms):
        body_cap = target_tokens if not windows else target_tokens - overlap_tokens
        body_cap = max(body_cap, 1)
        start, end = atoms[cursor]
        cursor += 1
        while cursor < len(atoms):
            following = atoms[cursor]
            if counter.count(text[start : following[1]]) > body_cap:
                break
            end = following[1]
            cursor += 1
        windows.append(
            _window_with_overlap(text, windows, start, end, counter, overlap_tokens)
        )
    return tuple(windows)


def build_source_chunks(
    entities: Sequence[AtlasEntity],
    edges: Sequence[AtlasRelationship],
    counter: TokenCounter,
    chunking: ChunkingConfig,
) -> tuple[SourceChunk, ...]:
    """Build every searchable source chunk in stable source order.

    Only technique, mitigation, and case-study descriptions plus relationship
    descriptions with text become passages.

    Args:
        entities (Sequence[AtlasEntity]): Normalized records in source order.
        edges (Sequence[AtlasRelationship]): Relationships in source order.
        counter (TokenCounter): Tokenizer used for every measurement.
        chunking (ChunkingConfig): Body and header budgets.

    Returns:
        (tuple[SourceChunk, ...]) Ordered searchable chunks.
    """
    by_id = {entity.entity_id: entity for entity in entities}
    chunks: list[SourceChunk] = []
    for entity in entities:
        if entity.kind not in _SEARCHABLE_KINDS:
            continue
        chunks.extend(_entity_chunks(entity, counter, chunking))
    for edge in edges:
        if (
            edge.relation not in _DESCRIPTION_RELATIONS
            or not (edge.description or "").strip()
        ):
            continue
        chunks.extend(_edge_chunks(edge, by_id, counter, chunking))
    return tuple(chunks)


def _entity_chunks(
    entity: AtlasEntity, counter: TokenCounter, chunking: ChunkingConfig
) -> list[SourceChunk]:
    """Chunk one searchable entity description."""
    label = f"{_KIND_LABELS[entity.kind]} {entity.entity_id}"
    extra = [f"Type: {entity.case_type}"] if entity.case_type else []
    header = _fit_header(label, entity.display_name, extra, counter, chunking)
    windows = chunk_windows(
        entity.description,
        counter,
        target_tokens=chunking.body_target_tokens,
        max_tokens=chunking.body_max_tokens,
        overlap_tokens=chunking.overlap_tokens,
    )
    return [
        SourceChunk(
            owner_kind=cast("OwnerKind", entity.kind),
            owner_id=entity.entity_id,
            entity_id=entity.entity_id,
            edge_id=None,
            section="description",
            ordinal=ordinal,
            source_locator=_chunk_locator(entity.source_locator, ordinal),
            header=header,
            text=window.text,
            start=window.start,
            end=window.end,
        )
        for ordinal, window in enumerate(windows)
    ]


def _edge_chunks(
    edge: AtlasRelationship,
    by_id: dict[str, AtlasEntity],
    counter: TokenCounter,
    chunking: ChunkingConfig,
) -> list[SourceChunk]:
    """Chunk one relationship description with its case and technique context."""
    owner = by_id.get(edge.source_id)
    target = by_id.get(edge.target_id)
    owner_kind = _owner_kind(owner)
    step = f" | Step {edge.step_id}" if edge.step_id else ""
    label = f"{_KIND_LABELS[owner_kind]} {edge.source_id}{step} | {edge.relation} {edge.target_id}"
    extra = []
    if target is not None:
        extra.append(f"Technique: {target.display_name}")
    if owner is not None and owner.case_type:
        extra.append(f"Type: {owner.case_type}")
    header = _fit_header(label, None, extra, counter, chunking)
    windows = chunk_windows(
        edge.description or "",
        counter,
        target_tokens=chunking.body_target_tokens,
        max_tokens=chunking.body_max_tokens,
        overlap_tokens=chunking.overlap_tokens,
    )
    return [
        SourceChunk(
            owner_kind=owner_kind,
            owner_id=edge.source_id,
            entity_id=None,
            edge_id=edge.edge_id,
            section=edge.relation,
            ordinal=ordinal,
            source_locator=_chunk_locator(edge.source_locator, ordinal),
            header=header,
            text=window.text,
            start=window.start,
            end=window.end,
        )
        for ordinal, window in enumerate(windows)
    ]


def _owner_kind(entity: AtlasEntity | None) -> OwnerKind:
    """Return the contextualizing owner kind of a relationship source."""
    if entity is None or entity.kind not in _SEARCHABLE_KINDS:
        return "case_study"
    return cast("OwnerKind", entity.kind)


def _chunk_locator(locator: str, ordinal: int) -> str:
    """Return the absolute locator of one chunk within its source field."""
    return locator if ordinal == 0 else f"{locator}/chunk/{ordinal}"


def _fit_header(
    label: str,
    title: str | None,
    extra: Sequence[str],
    counter: TokenCounter,
    chunking: ChunkingConfig,
) -> str:
    """Fit a header to the token budget without truncating an identifier."""
    limit = chunking.header_max_tokens
    if title:
        words = title.split()
        for count in range(len(words), 0, -1):
            candidate = _join_header(f"{label} | {' '.join(words[:count])}...", extra)
            if counter.count(candidate) <= limit:
                return candidate
        abbreviated = _join_header(f"{label} | ...", extra)
        if counter.count(abbreviated) <= limit:
            return abbreviated
    candidate = _join_header(label, extra)
    if counter.count(candidate) <= limit:
        return candidate
    return label


def _join_header(headline: str, extra: Sequence[str]) -> str:
    """Join the headline and its metadata lines."""
    return "\n".join([headline, *extra])


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """Return trimmed paragraph spans in source order."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return _trimmed(text, spans)


def _sentences(text: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    """Return trimmed sentence spans inside one paragraph span."""
    start, end = span
    matches = [
        (start + match.start(), start + match.end())
        for match in _SENTENCE_BREAK.finditer(text[start:end])
    ]
    return _trimmed(text, matches)


def _trimmed(text: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop blank spans and strip surrounding whitespace from the rest."""
    trimmed: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        low, high = span_start, span_end
        while low < high and text[low].isspace():
            low += 1
        while high > low and text[high - 1].isspace():
            high -= 1
        if high > low:
            trimmed.append((low, high))
    return trimmed


def _split_oversized(
    text: str,
    spans: list[tuple[int, int]],
    counter: TokenCounter,
    limit: int,
) -> list[tuple[int, int]]:
    """Split spans that exceed the hard limit into sentence or token pieces."""
    pieces: list[tuple[int, int]] = []
    for span in spans:
        pieces.extend(_split_span(text, span, counter, limit))
    return pieces


def _split_span(
    text: str,
    span: tuple[int, int],
    counter: TokenCounter,
    limit: int,
) -> list[tuple[int, int]]:
    """Split one span until every piece fits, always making progress."""
    if counter.count(text[span[0] : span[1]]) <= limit:
        return [span]
    sentences = _sentences(text, span)
    if len(sentences) > 1:
        return _split_oversized(text, sentences, counter, limit)
    return _slice_span(text, span, counter, limit)


def _slice_span(
    text: str,
    span: tuple[int, int],
    counter: TokenCounter,
    limit: int,
) -> list[tuple[int, int]]:
    """Slice one over-long span at token boundaries until every piece fits."""
    pieces: list[tuple[int, int]] = []
    start, end = span
    while start < end:
        offset = counter.prefix_offset(text[start:end], limit)
        if offset <= 0:
            offset = end - start
        piece_end = _strip_end(text, start, start + offset)
        if piece_end <= start:
            piece_end = start + offset
        pieces.append((start, piece_end))
        start = piece_end
        while start < end and text[start].isspace():
            start += 1
    return pieces


def _strip_end(text: str, start: int, end: int) -> int:
    """Move an end offset back over trailing whitespace."""
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _window_with_overlap(
    text: str,
    windows: Sequence[TextWindow],
    start: int,
    end: int,
    counter: TokenCounter,
    overlap_tokens: int,
) -> TextWindow:
    """Extend a window forward over the previous window's trailing tokens."""
    if windows and overlap_tokens:
        previous = windows[-1]
        start = previous.start + counter.tail_offset(previous.text, overlap_tokens)
    return TextWindow(start=start, end=end, text=text[start:end])
