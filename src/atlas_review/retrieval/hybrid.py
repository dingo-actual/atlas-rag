"""Dense and BM25 initial search channels with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from atlas_review.config import AppConfig
from atlas_review.contracts import Candidate, Passage, SearchChannel
from atlas_review.errors import STAGE_RETRIEVAL, AtlasReviewError
from atlas_review.retrieval.embeddings import EmbeddingBackend, query_vector
from atlas_review.retrieval.keyword_index import keyword_search
from atlas_review.retrieval.loader import LoadedGeneration
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.retrieval.vector_index import dense_search

Hits = tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Result of one local retrieval run over the initial search channels.

    Instance Attributes:
        generation_id (str): Generation searched.
        collection_version (str): ATLAS collection version searched.
        focus (str): Retrieval focus used for both channels.
        dense_hits (Hits): ``(passage_id, distance)`` in dense order.
        keyword_hits (Hits): ``(passage_id, score)`` in BM25 order.
        keyword_zero_score_exclusions (int): Dropped zero-score keyword results.
        candidates (tuple[Candidate, ...]): Fused candidates, best first.
        passages (tuple[Passage, ...]): Passages in fused candidate order.
    """

    generation_id: str
    collection_version: str
    focus: str
    dense_hits: Hits
    keyword_hits: Hits
    keyword_zero_score_exclusions: int
    candidates: tuple[Candidate, ...]
    passages: tuple[Passage, ...]


def check_focus(focus: str, counters: TokenCounters, config: AppConfig) -> int:
    """Validate the retrieval focus against the query token budget.

    Args:
        focus (str): Ordinary retrieval focus.
        counters (TokenCounters): Retrieval tokenizers.
        config (AppConfig): Retrieval limits.

    Returns:
        (int) Reranker token count of the focus.

    Raises:
        AtlasReviewError: The focus is blank or longer than the budget.
    """
    if not focus.strip():
        raise _retrieval_error("blank_focus", "the retrieval focus is empty")
    tokens = max(counters.reranker.count(focus), counters.embedding.count(focus))
    if tokens > config.retrieval.query_max_tokens:
        raise _retrieval_error(
            "query_too_long",
            f"the retrieval focus uses {tokens} tokens; "
            f"the limit is {config.retrieval.query_max_tokens}",
        )
    return tokens


def search(
    loaded: LoadedGeneration,
    backend: EmbeddingBackend,
    config: AppConfig,
    focus: str,
) -> SearchOutcome:
    """Run dense and BM25 search and fuse their ranks.

    Args:
        loaded (LoadedGeneration): Verified indexes and passages.
        backend (EmbeddingBackend): Embedding implementation.
        config (AppConfig): Retrieval and index settings.
        focus (str): Ordinary retrieval focus.

    Returns:
        (SearchOutcome) Channel hits, fused candidates, and passages.

    Raises:
        AtlasReviewError: A channel returns a passage outside the generation.
    """
    vector = query_vector(backend, config.embedding, focus)
    dense_hits = dense_search(
        loaded.dense_index,
        vector,
        count=config.retrieval.dense_k,
        threads=config.vector_index.threads,
    )
    keyword = keyword_search(
        loaded.keyword_index,
        focus,
        limit=config.retrieval.bm25_k,
        passage_ids_by_ordinal=loaded.passage_ids,
        config=config.keyword_index,
    )
    keyword_hits = keyword.hits
    known = set(loaded.passages)
    for passage_id, _ in (*dense_hits, *keyword_hits):
        if passage_id not in known:
            raise _retrieval_error(
                "foreign_index_key",
                f"index hit {passage_id} is not a passage of this generation",
            )
    candidates = fuse(dense_hits, keyword_hits, rrf_k=config.retrieval.fusion_rrf_k)
    return SearchOutcome(
        generation_id=loaded.generation_id,
        collection_version=loaded.collection_version,
        focus=focus,
        dense_hits=dense_hits,
        keyword_hits=keyword_hits,
        keyword_zero_score_exclusions=keyword.zero_score_exclusions,
        candidates=candidates,
        passages=tuple(
            loaded.passages[candidate.passage_id] for candidate in candidates
        ),
    )


def fuse(dense_hits: Hits, keyword_hits: Hits, *, rrf_k: int) -> tuple[Candidate, ...]:
    """Fuse two ranked lists with reciprocal rank fusion.

    Args:
        dense_hits (Hits): Dense hits in dense order.
        keyword_hits (Hits): BM25 hits in score order.
        rrf_k (int): Fusion constant; must be positive.

    Returns:
        (tuple[Candidate, ...]) Candidates ordered by fused score, then ID.

    Raises:
        AtlasReviewError: The fusion constant is not positive.
    """
    if rrf_k <= 0:
        raise _retrieval_error("invalid_rrf_constant", "fusion_rrf_k must be positive")
    dense_rank = {
        passage_id: position for position, (passage_id, _) in enumerate(dense_hits, 1)
    }
    dense_distance = dict(dense_hits)
    keyword_rank = {
        passage_id: position for position, (passage_id, _) in enumerate(keyword_hits, 1)
    }
    keyword_score = dict(keyword_hits)
    candidates: list[Candidate] = []
    for passage_id in set(dense_rank) | set(keyword_rank):
        score = 0.0
        if passage_id in dense_rank:
            score += 1.0 / (rrf_k + dense_rank[passage_id])
        if passage_id in keyword_rank:
            score += 1.0 / (rrf_k + keyword_rank[passage_id])
        channels: list[SearchChannel] = []
        if passage_id in dense_rank:
            channels.append("dense")
        if passage_id in keyword_rank:
            channels.append("bm25")
        candidates.append(
            Candidate(
                passage_id=passage_id,
                channels=channels,
                dense_rank=dense_rank.get(passage_id),
                dense_distance=dense_distance.get(passage_id),
                bm25_rank=keyword_rank.get(passage_id),
                bm25_score=keyword_score.get(passage_id),
                fusion_score=score,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (-candidate.fusion_score, candidate.passage_id),
        )
    )


def _retrieval_error(code: str, message: str) -> AtlasReviewError:
    """Build a retrieval-input error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_RETRIEVAL,
        message=message,
        recovery="Shorten the retrieval focus and retrieve again.",
    )
