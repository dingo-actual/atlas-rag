"""Normalization tests: conservative visible-text handling and deterministic names."""

import pytest

from atlas_review.contracts import AtlasEntity, AtlasRelationship, SourceReference
from atlas_review.corpus.normalize import (
    derive_display_names,
    normalize_optional,
    normalize_source_text,
    resolve_reference_markers,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("  a   b\t c  ", "a b c", id="collapse-spaces"),
        pytest.param("line one\r\nline two", "line one\nline two", id="crlf"),
        pytest.param("one\n\n\n\ntwo", "one\n\ntwo", id="paragraphs"),
        pytest.param("\n\n  padded  \n\n", "padded", id="trim-edges"),
        pytest.param("a &amp; b &lt; c", "a & b < c", id="entities"),
        pytest.param(
            "text <code>IGNORE</code> rest", "text IGNORE rest", id="inline-tag-text"
        ),
        pytest.param(
            "payload <system>reset</system> end",
            "payload reset end",
            id="attack-like-markup",
        ),
        pytest.param("keep <br> break", "keep\nbreak", id="line-break-tag"),
        pytest.param(
            "visible<script>hidden()</script> end", "visible end", id="drop-script"
        ),
        pytest.param(
            "bell\x07 and \x1b[31mred", "bell and [31mred", id="control-chars"
        ),
        pytest.param("a  b\nc  d", "a b\nc d", id="per-line-collapse"),
    ],
)
def test_normalize_source_text(raw: str, expected: str) -> None:
    assert normalize_source_text(raw) == expected


def test_normalization_is_idempotent() -> None:
    raw = "  <p>One &amp; two</p>\n\n\nthree   four  "
    once = normalize_source_text(raw)
    assert normalize_source_text(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param("   \n\t ", None, id="blank"),
        pytest.param(" value ", "value", id="value"),
    ],
)
def test_normalize_optional(raw: str | None, expected: str | None) -> None:
    assert normalize_optional(raw) == expected


def build_references() -> list[SourceReference]:
    return [
        SourceReference(ref_id="with-url", title="Report", url="https://example.com/r"),
        SourceReference(ref_id="no-url", title="Offline Report"),
    ]


@pytest.mark.parametrize(
    ("text", "expected", "warnings"),
    [
        pytest.param(
            "see [[with-url]].", "see [Report](https://example.com/r).", [], id="link"
        ),
        pytest.param("see [[no-url]].", "see [Offline Report].", [], id="no-url"),
        pytest.param(
            "see [[unknown]].",
            "see [[unknown]].",
            ["unresolved bibliography marker [[unknown]]"],
            id="unresolved",
        ),
        pytest.param("no markers", "no markers", [], id="plain"),
    ],
)
def test_resolve_reference_markers(
    text: str, expected: str, warnings: list[str]
) -> None:
    resolved, produced = resolve_reference_markers(text, build_references())
    assert resolved == expected
    assert produced == warnings


def entity(entity_id: str, name: str) -> AtlasEntity:
    return AtlasEntity(
        generation_id="generation-1",
        entity_id=entity_id,
        kind="technique",
        name=name,
        display_name=name,
        description=f"{name} description",
        source_locator=f"/techniques/{entity_id}/description",
    )


def specializes(child: str, parent: str) -> AtlasRelationship:
    return AtlasRelationship(
        generation_id="generation-1",
        edge_id=("0" * 63) + str(len(child) % 10),
        relation="specializes",
        source_id=child,
        target_id=parent,
        source_locator=f"/relationships/{child}/specializes/0/description",
    )


def test_display_names_use_parent_chain_deterministically() -> None:
    entities = [
        entity("AML.T0051", "LLM Prompt Injection"),
        entity("AML.T0051.001", "Indirect"),
        entity("AML.T0051.001.001", "Nested"),
    ]
    edges = [
        specializes("AML.T0051.001", "AML.T0051"),
        specializes("AML.T0051.001.001", "AML.T0051.001"),
    ]
    names = derive_display_names(entities, edges)
    assert names["AML.T0051"] == "LLM Prompt Injection"
    assert names["AML.T0051.001"] == "LLM Prompt Injection / Indirect"
    assert names["AML.T0051.001.001"] == "LLM Prompt Injection / Indirect / Nested"


def test_display_names_ignore_non_specializes_edges() -> None:
    entities = [
        entity("AML.T0051", "LLM Prompt Injection"),
        entity("AML.CS0035", "Case"),
    ]
    edges = [
        AtlasRelationship(
            generation_id="generation-1",
            edge_id="a" * 64,
            relation="achieves",
            source_id="AML.T0051",
            target_id="AML.CS0035",
            source_locator="/relationships/AML.T0051/achieves/0/description",
        )
    ]
    names = derive_display_names(entities, edges)
    assert names["AML.CS0035"] == "Case"
