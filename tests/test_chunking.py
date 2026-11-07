"""Token-aware chunking: budget, overlap, offsets, and source chunk assembly."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from atlas_review.config import ChunkingConfig
from atlas_review.contracts import AtlasEntity
from atlas_review.corpus.atlas_v6 import ParsedCorpus, parse_source
from atlas_review.corpus.chunk import (
    SourceChunk,
    TextWindow,
    build_source_chunks,
    chunk_windows,
)
from fakes import WhitespaceTokenCounter

COUNTER = WhitespaceTokenCounter()


def paragraphs(*counts: int) -> str:
    """Return paragraphs of the given word counts separated by blank lines."""
    return "\n\n".join(
        " ".join(f"word{index}" for index in range(count)) for count in counts
    )


def tokens(text: str) -> list[str]:
    """Return whitespace tokens for order comparisons."""
    return text.split()


class TestChunkWindows:
    """Every window stays inside the budget and advances through the text."""

    def test_blank_text_produces_no_windows(self) -> None:
        assert (
            chunk_windows(
                "  \n\n ", COUNTER, target_tokens=10, max_tokens=10, overlap_tokens=0
            )
            == ()
        )

    def test_short_text_produces_one_window_covering_everything(self) -> None:
        text = paragraphs(5, 5)
        windows = chunk_windows(
            text, COUNTER, target_tokens=20, max_tokens=20, overlap_tokens=0
        )
        assert windows == (TextWindow(start=0, end=len(text), text=text),)

    def test_paragraphs_are_packed_to_the_token_target(self) -> None:
        text = paragraphs(10, 10, 10, 10)
        windows = chunk_windows(
            text, COUNTER, target_tokens=20, max_tokens=20, overlap_tokens=0
        )
        assert [len(tokens(window.text)) for window in windows] == [20, 20]
        assert tokens(" ".join(window.text for window in windows)) == tokens(text)

    def test_overlap_repeats_the_previous_tail_without_exceeding_the_budget(
        self,
    ) -> None:
        text = paragraphs(10, 10, 10, 10)
        windows = chunk_windows(
            text, COUNTER, target_tokens=20, max_tokens=20, overlap_tokens=5
        )
        assert len(windows) == 3
        assert all(len(tokens(window.text)) <= 20 for window in windows)
        for previous, current in pairwise(windows):
            assert tokens(current.text)[:5] == tokens(previous.text)[-5:]

    def test_oversized_paragraph_is_sliced_without_losing_text(self) -> None:
        text = " ".join(f"word{index}" for index in range(100))
        windows = chunk_windows(
            text, COUNTER, target_tokens=20, max_tokens=20, overlap_tokens=3
        )
        assert len(windows) > 1
        assert all(len(tokens(window.text)) <= 20 for window in windows)
        assert windows[0].start == 0
        assert windows[-1].end == len(text)
        assert all(window.text == text[window.start : window.end] for window in windows)

    @pytest.mark.parametrize(
        ("target", "maximum", "overlap"),
        [
            pytest.param(0, 10, 0, id="zero-target"),
            pytest.param(10, 0, 0, id="zero-maximum"),
            pytest.param(10, 10, 10, id="overlap-fills-the-budget"),
            pytest.param(10, 10, -1, id="negative-overlap"),
        ],
    )
    def test_invalid_budgets_are_rejected(
        self, target: int, maximum: int, overlap: int
    ) -> None:
        with pytest.raises(ValueError, match="token"):
            chunk_windows(
                "some text",
                COUNTER,
                target_tokens=target,
                max_tokens=maximum,
                overlap_tokens=overlap,
            )

    @given(
        st.lists(
            st.text(alphabet="abc", min_size=1, max_size=6), min_size=1, max_size=12
        )
    )
    def test_windows_always_advance_and_cover_the_text(self, words: list[str]) -> None:
        text = "\n\n".join(" ".join(words[index:]) for index in range(0, len(words), 3))
        windows = chunk_windows(
            text, COUNTER, target_tokens=8, max_tokens=12, overlap_tokens=2
        )
        assert windows
        assert windows[0].start == 0
        assert windows[-1].end == len(text)
        assert all(window.text == text[window.start : window.end] for window in windows)
        assert all(len(tokens(window.text)) <= 12 for window in windows)
        starts = [window.start for window in windows]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)


@pytest.fixture
def fixture_corpus(source_path: Path) -> ParsedCorpus:
    """Parse the committed fixture once for the chunk-assembly tests."""
    return parse_source(
        source_path.read_bytes(),
        origin=str(source_path),
        acquired_at="2026-11-01T00:00:00+00:00",
    )


class TestSourceChunks:
    """Searchable passages come only from the documented source sections."""

    def test_only_documented_sections_become_chunks(
        self, fixture_corpus: ParsedCorpus
    ) -> None:
        corpus = fixture_corpus
        chunks = build_source_chunks(
            corpus.entities, corpus.edges, COUNTER, ChunkingConfig()
        )
        owners = {(chunk.entity_id, chunk.edge_id, chunk.section) for chunk in chunks}
        assert ("AML.T0051", None, "description") in owners
        assert ("AML.M0033", None, "description") in owners
        assert ("AML.CS0035", None, "description") in owners
        assert (None, _edge_id(corpus, "S01"), "employs") in owners
        assert (None, _edge_id(corpus, "mitigates"), "mitigates") in owners
        assert "AML.TA0005" not in {chunk.owner_id for chunk in chunks}

    def test_chunk_headers_preserve_identifiers_and_case_type(
        self, fixture_corpus: ParsedCorpus
    ) -> None:
        corpus = fixture_corpus
        chunks = build_source_chunks(
            corpus.entities, corpus.edges, COUNTER, ChunkingConfig()
        )
        technique = _chunk_for(chunks, "AML.T0051.001", None)
        assert technique.header.startswith("Technique AML.T0051.001 | ")
        assert "LLM Prompt Injection / Indirect" in technique.header
        case = _chunk_for(chunks, "AML.CS0035", None)
        assert "AML.CS0035" in case.header
        assert "Type: Exercise" in case.header
        step = _chunk_for(chunks, None, _edge_id(corpus, "S01"))
        assert "AML.CS0035" in step.header
        assert "Step S01" in step.header
        assert "employs AML.T0051.001" in step.header

    def test_relationship_chunks_keep_their_source_locator(
        self, fixture_corpus: ParsedCorpus
    ) -> None:
        corpus = fixture_corpus
        chunks = build_source_chunks(
            corpus.entities, corpus.edges, COUNTER, ChunkingConfig()
        )
        step = _chunk_for(chunks, None, _edge_id(corpus, "S01"))
        assert step.source_locator.startswith("/relationships/AML.CS0035/employs/")
        assert step.text

    def test_long_titles_are_abbreviated_but_identifiers_survive(self) -> None:
        entity = AtlasEntity(
            generation_id="generation-1",
            entity_id="AML.T9999",
            kind="technique",
            name="A deliberately very long technique name that cannot fit",
            display_name="A deliberately very long technique name",
            description="A short description of the synthetic technique.",
            source_locator="/techniques/AML.T9999/description",
        )
        chunks = build_source_chunks(
            (entity,), (), COUNTER, ChunkingConfig(header_max_tokens=6)
        )
        assert len(chunks) == 1
        assert COUNTER.count(chunks[0].header) <= 6
        assert "AML.T9999" in chunks[0].header
        assert "..." in chunks[0].header


def _edge_id(corpus: ParsedCorpus, marker: str) -> str:
    """Return the edge identifier a fixture relationship marker belongs to."""
    for edge in corpus.edges:
        if marker in (edge.step_id, edge.relation):
            return edge.edge_id
    raise AssertionError(f"no fixture edge matches {marker}")


def _chunk_for(
    chunks: tuple[SourceChunk, ...], entity_id: str | None, edge_id: str | None
) -> SourceChunk:
    """Return the single fixture chunk for one owner."""
    matches = [
        chunk
        for chunk in chunks
        if chunk.entity_id == entity_id and chunk.edge_id == edge_id
    ]
    assert len(matches) == 1
    return matches[0]
