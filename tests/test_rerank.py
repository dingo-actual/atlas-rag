"""Cross-encoder reranking: alignment, ordering, and score validation."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from atlas_review.config import AppConfig
from atlas_review.contracts import Candidate, Passage
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.rerank import RERANKER_PAIR_LIMIT, rerank
from atlas_review.retrieval.tokenizers import TokenCounters
from fakes import RecordingReranker, WhitespaceTokenCounter

DIGEST = "a" * 64


def _passage(passage_id: int, retrieval_text: str) -> Passage:
    """Build a minimal validated passage for reranking tests."""
    return Passage(
        schema_version=1,
        passage_id=passage_id,
        generation_id="gen",
        entity_id="AML.T0051",
        owner_kind="technique",
        section="description",
        ordinal=0,
        bm25_ordinal=passage_id - 1,
        header="Technique AML.T0051 | LLM Prompt Injection",
        source_text=retrieval_text,
        context_text="generated note",
        retrieval_text=retrieval_text,
        context_cache_key=DIGEST,
        source_sha256=DIGEST,
        content_sha256=DIGEST,
        embedding_tokens=1,
        reranker_tokens=1,
        source_locator="/techniques/AML.T0051/description",
        source_start=0,
        source_end=1,
    )


def _candidate(passage_id: int) -> Candidate:
    """Build a dense-only candidate for the given passage."""
    return Candidate(
        passage_id=passage_id,
        channels=["dense"],
        dense_rank=passage_id,
        dense_distance=0.5,
        fusion_score=0.01,
    )


@pytest.fixture
def counters() -> TokenCounters:
    """Return whitespace counters for both retrieval tokenizers."""
    return TokenCounters(
        embedding=WhitespaceTokenCounter(), reranker=WhitespaceTokenCounter()
    )


def _rerank(
    candidates: list[Candidate],
    passages: dict[int, Passage],
    reranker: RecordingReranker,
    counters: TokenCounters,
    focus: str = "focus",
) -> tuple[Candidate, ...]:
    """Call the adapter with the test's validated configuration."""
    return rerank(
        candidates, passages, reranker, focus, counters=counters, config=AppConfig()
    )


class TestRerankOrdering:
    """Scores stay aligned with their documents and order the candidates."""

    def test_higher_scores_rank_first_and_ties_break_by_passage_id(
        self, counters: TokenCounters
    ) -> None:
        candidates = [_candidate(7), _candidate(5), _candidate(6)]
        passages = {number: _passage(number, "text") for number in (5, 6, 7)}
        ranked = _rerank(
            candidates,
            passages,
            RecordingReranker(scores=(0.1, 0.5, 0.5)),
            counters,
        )
        assert [candidate.passage_id for candidate in ranked] == [5, 6, 7]
        assert [candidate.reranker_score for candidate in ranked] == [0.5, 0.5, 0.1]

    def test_reranker_sees_augmented_texts_in_candidate_order(
        self, counters: TokenCounters
    ) -> None:
        candidates = [_candidate(2), _candidate(1)]
        passages = {1: _passage(1, "first passage"), 2: _passage(2, "second passage")}
        reranker = RecordingReranker()
        _rerank(candidates, passages, reranker, counters, focus="ordinary focus")
        assert reranker.calls == [
            ("ordinary focus", ("second passage", "first passage"))
        ]

    def test_empty_candidate_list_needs_no_reranker_call(
        self, counters: TokenCounters
    ) -> None:
        reranker = RecordingReranker()
        assert _rerank([], {}, reranker, counters) == ()
        assert reranker.calls == []


class TestRerankValidation:
    """Invalid scores, counts, and oversized pairs fail the retrieval run."""

    def test_score_count_mismatch_is_rejected(self, counters: TokenCounters) -> None:
        candidates = [_candidate(1), _candidate(2)]
        passages = {1: _passage(1, "a"), 2: _passage(2, "b")}
        with pytest.raises(AtlasReviewError) as caught:
            _rerank(candidates, passages, RecordingReranker(scores=(0.2,)), counters)
        assert caught.value.code == "reranker_count_mismatch"

    @pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
    def test_non_finite_score_is_rejected(
        self, counters: TokenCounters, score: float
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            _rerank(
                [_candidate(1)],
                {1: _passage(1, "a")},
                RecordingReranker(scores=(score,)),
                counters,
            )
        assert caught.value.code == "invalid_reranker_score"

    def test_pair_above_the_model_limit_is_rejected(
        self, counters: TokenCounters
    ) -> None:
        query = " ".join(["word"] * 12)
        document = " ".join(["term"] * (RERANKER_PAIR_LIMIT - 8))
        with pytest.raises(AtlasReviewError) as caught:
            _rerank(
                [_candidate(1)],
                {1: _passage(1, document)},
                RecordingReranker(),
                counters,
                focus=query,
            )
        assert caught.value.code == "reranker_pair_too_long"


@given(
    scores=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        min_size=1,
        max_size=8,
        unique=True,
    )
)
def test_ranking_is_a_descending_permutation_of_the_scores(
    scores: list[float],
) -> None:
    counters = TokenCounters(
        embedding=WhitespaceTokenCounter(), reranker=WhitespaceTokenCounter()
    )
    identifiers = list(range(1, len(scores) + 1))
    passages = {number: _passage(number, "text") for number in identifiers}
    ranked = _rerank(
        [_candidate(number) for number in identifiers],
        passages,
        RecordingReranker(scores=tuple(scores)),
        counters,
    )
    assert [candidate.reranker_score for candidate in ranked] == sorted(
        scores, reverse=True
    )
