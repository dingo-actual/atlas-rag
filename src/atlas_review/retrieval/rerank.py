"""Cross-encoder reranking of the capped candidate pool.

The reranker scores the same augmented retrieval text used by both indexes and
the ordinary retrieval focus without the BGE query instruction. Scores order
evidence; they are model-specific relevance values, not probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from atlas_review.config import AppConfig, RerankerConfig
from atlas_review.contracts import Candidate, Passage
from atlas_review.errors import STAGE_RETRIEVAL, AtlasReviewError
from atlas_review.retrieval.tokenizers import TokenCounters

RERANKER_PAIR_LIMIT = 512


class Reranker(Protocol):
    """Score a query against documents, one score per document in order."""

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """Return one relevance score per document, in input order."""
        ...


class FastEmbedReranker:
    """Score query-passage pairs with the fixed FastEmbed cross-encoder.

    Instance Attributes:
        _config (RerankerConfig): Fixed model, batch, and thread settings.
        _model (TextCrossEncoder): Lazily reusable reranking session.
    """

    def __init__(self, config: RerankerConfig, cache_dir: Path) -> None:
        """Load or download the configured cross-encoder.

        Args:
            config (RerankerConfig): Fixed model, batch, and thread settings.
            cache_dir (Path): Workspace model cache directory.
        """
        self._config = config
        self._model = TextCrossEncoder(
            model_name=config.model_name,
            cache_dir=str(cache_dir),
            threads=config.threads,
            providers=[config.provider],
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, in input order."""
        return [
            float(value)
            for value in self._model.rerank(
                query, list(documents), batch_size=self._config.batch_size
            )
        ]


def rerank(
    candidates: Sequence[Candidate],
    passages: Mapping[int, Passage],
    reranker: Reranker,
    focus: str,
    *,
    counters: TokenCounters,
    config: AppConfig,
) -> tuple[Candidate, ...]:
    """Score the capped pool and order it by relevance, then passage ID.

    Args:
        candidates (Sequence[Candidate]): Candidate pool in admission order.
        passages (Mapping[int, Passage]): Stored passages by identifier.
        reranker (Reranker): Cross-encoder implementation.
        focus (str): Ordinary retrieval focus without the BGE instruction.
        counters (TokenCounters): Retrieval tokenizers used for the pair check.
        config (AppConfig): Retrieval and reranker settings.

    Returns:
        (tuple[Candidate, ...]) Reranked candidates carrying their score.

    Raises:
        AtlasReviewError: The pool is oversized, a pair exceeds the model
            input budget, or the returned scores are unusable.
    """
    if len(candidates) > config.retrieval.rerank_candidate_limit:
        raise _rerank_error(
            "rerank_pool_too_large",
            f"the candidate pool has {len(candidates)} passages; "
            f"the reranker accepts {config.retrieval.rerank_candidate_limit}",
        )
    if not candidates:
        return ()
    documents = []
    for candidate in candidates:
        passage = passages.get(candidate.passage_id)
        if passage is None:
            raise _rerank_error(
                "unknown_candidate_passage",
                f"candidate {candidate.passage_id} has no stored passage",
            )
        pair_tokens = counters.reranker.count_pair(focus, passage.retrieval_text)
        if pair_tokens > RERANKER_PAIR_LIMIT:
            raise _rerank_error(
                "reranker_pair_too_long",
                f"the query-passage pair uses {pair_tokens} tokens; "
                f"the reranker limit is {RERANKER_PAIR_LIMIT}",
            )
        documents.append(passage.retrieval_text)
    scores = list(reranker.score(focus, documents))
    if len(scores) != len(candidates):
        raise _rerank_error(
            "reranker_count_mismatch",
            f"the reranker returned {len(scores)} scores for "
            f"{len(candidates)} passages",
        )
    if any(not math.isfinite(score) for score in scores):
        raise _rerank_error(
            "invalid_reranker_score", "the reranker returned a non-finite score"
        )
    return tuple(
        sorted(
            (
                _with_score(candidate, score)
                for candidate, score in zip(candidates, scores, strict=True)
            ),
            key=lambda candidate: (-_score(candidate), candidate.passage_id),
        )
    )


def _with_score(candidate: Candidate, score: float) -> Candidate:
    """Return a validated candidate copy carrying its reranker score."""
    return Candidate.model_validate(
        {**candidate.model_dump(mode="json"), "reranker_score": float(score)},
        strict=True,
    )


def _score(candidate: Candidate) -> float:
    """Return the recorded reranker score of one ranked candidate."""
    return candidate.reranker_score or 0.0


def _rerank_error(code: str, message: str) -> AtlasReviewError:
    """Build a reranking-stage error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_RETRIEVAL,
        message=message,
        recovery=(
            "Retrieve again with a shorter focus, or repair the generation's "
            "index bundle before reranking."
        ),
    )
