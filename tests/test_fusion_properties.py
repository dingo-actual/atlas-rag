"""Property tests for reciprocal rank fusion and the passage-ID mapping."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from atlas_review.config import AppConfig
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.artifacts import (
    decode_passage_ids_payload,
    encode_passage_ids,
)
from atlas_review.retrieval.hybrid import fuse

RRF_K = AppConfig().retrieval.fusion_rrf_k
passage_ids = st.integers(min_value=1, max_value=2**31)
hits = st.lists(
    st.tuples(passage_ids, st.floats(min_value=0.0, max_value=1000.0)),
    max_size=12,
)


def ranked(hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Return hits keyed once each, in rank order."""
    seen: set[int] = set()
    unique: list[tuple[int, float]] = []
    for passage_id, value in hits:
        if passage_id not in seen:
            seen.add(passage_id)
            unique.append((passage_id, value))
    return unique


class TestFusion:
    """Fusion is total, positive, and ordered by channel ranks."""

    @given(hits, hits)
    def test_every_channel_hit_appears_exactly_once(
        self, dense: list[tuple[int, float]], keyword: list[tuple[int, float]]
    ) -> None:
        dense_hits = tuple(ranked(dense))
        keyword_hits = tuple(ranked(keyword))
        candidates = fuse(dense_hits, keyword_hits, rrf_k=RRF_K)
        found = [candidate.passage_id for candidate in candidates]
        assert len(found) == len(set(found))
        assert set(found) == {key for key, _ in dense_hits} | {
            key for key, _ in keyword_hits
        }

    @given(hits, hits)
    def test_scores_are_positive_and_ordered(
        self, dense: list[tuple[int, float]], keyword: list[tuple[int, float]]
    ) -> None:
        candidates = fuse(tuple(ranked(dense)), tuple(ranked(keyword)), rrf_k=RRF_K)
        scores = [candidate.fusion_score for candidate in candidates]
        assert all(score > 0.0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    @given(hits)
    def test_single_channel_scores_follow_reciprocal_rank(
        self, dense: list[tuple[int, float]]
    ) -> None:
        dense_hits = tuple(ranked(dense))
        candidates = fuse(dense_hits, (), rrf_k=RRF_K)
        expected = {
            passage_id: 1.0 / (RRF_K + rank)
            for rank, (passage_id, _) in enumerate(dense_hits, start=1)
        }
        for candidate in candidates:
            assert candidate.fusion_score == pytest.approx(
                expected[candidate.passage_id]
            )
            assert candidate.channels == ["dense"]

    def test_shared_hit_outranks_either_channel_alone(self) -> None:
        candidates = fuse(((7, 0.1), (8, 0.2)), ((9, 2.0), (7, 1.0)), rrf_k=RRF_K)
        assert candidates[0].passage_id == 7
        assert candidates[0].channels == ["dense", "bm25"]

    def test_zero_fusion_constant_is_rejected(self) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            fuse(((7, 0.1),), (), rrf_k=0)
        assert caught.value.code == "invalid_rrf_constant"


class TestPassageIdMapping:
    """The stored ordinal mapping round-trips without changing values."""

    @given(st.lists(passage_ids, min_size=1, max_size=32, unique=True))
    def test_round_trip_preserves_order(self, ids: list[int]) -> None:
        assert decode_passage_ids_payload(encode_passage_ids(ids)) == tuple(ids)

    @given(st.lists(st.integers(min_value=-1000, max_value=0), min_size=1))
    def test_non_positive_ids_are_rejected(self, ids: list[int]) -> None:
        with pytest.raises(ValidationError):
            encode_passage_ids(ids)

    def test_corrupt_payload_is_rejected(self) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            decode_passage_ids_payload(b"\xc1\xc1\xc1")
        assert caught.value.code == "invalid_keyword_mapping"
