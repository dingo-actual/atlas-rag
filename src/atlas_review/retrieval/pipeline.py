"""One local retrieval run: initial channels, graph expansion, reranking, evidence.

The run freezes every stage in a validated trace. No provider request happens
here: generated context was produced during ingestion and is used only as the
augmented retrieval text the reranker scores.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from atlas_review.config import AppConfig
from atlas_review.contracts import (
    Candidate,
    ChannelHit,
    RetrievalTrace,
    StageTiming,
)
from atlas_review.hashing import fingerprint
from atlas_review.retrieval.embeddings import EmbeddingBackend
from atlas_review.retrieval.evidence import Pool, Selection, build_pool, select_evidence
from atlas_review.retrieval.graph_search import GraphExpansion, expand_graph
from atlas_review.retrieval.hybrid import SearchOutcome, check_focus, search
from atlas_review.retrieval.loader import LoadedGeneration
from atlas_review.retrieval.rerank import Reranker, rerank
from atlas_review.retrieval.tokenizers import TokenCounters


@dataclass(frozen=True, slots=True)
class RetrievalRun:
    """Everything one local retrieval run produced.

    Instance Attributes:
        outcome (SearchOutcome): Dense, BM25, and fused direct results.
        expansion (GraphExpansion): Seeds, paths, and graph-derived passages.
        pool (Pool): Capped candidate pool passed to the reranker.
        ranked (tuple[Candidate, ...]): Reranked pool in evidence order.
        selection (Selection): Frozen packet, exclusions, and gaps.
        trace (RetrievalTrace): Validated diagnostic record of the run.
    """

    outcome: SearchOutcome
    expansion: GraphExpansion
    pool: Pool
    ranked: tuple[Candidate, ...]
    selection: Selection
    trace: RetrievalTrace


def retrieve(
    loaded: LoadedGeneration,
    backend: EmbeddingBackend,
    reranker: Reranker,
    config: AppConfig,
    focus: str,
    counters: TokenCounters,
    *,
    model_load_seconds: float = 0.0,
) -> RetrievalRun:
    """Run the full local retrieval pipeline over one loaded generation.

    Args:
        loaded (LoadedGeneration): Verified indexes, passages, and graph.
        backend (EmbeddingBackend): Embedding implementation.
        reranker (Reranker): Cross-encoder implementation.
        config (AppConfig): Retrieval, index, and reranker settings.
        focus (str): Ordinary retrieval focus without the BGE instruction.
        counters (TokenCounters): Retrieval tokenizers for budget checks.
        model_load_seconds (float): Time already spent acquiring model artifacts.

    Returns:
        (RetrievalRun) Traceable retrieval result with a frozen packet.

    Raises:
        AtlasReviewError: A budget, explicit identifier, index, or reranker
            check fails; no reduced pipeline is substituted.
    """
    query_tokens = check_focus(focus, counters, config)
    started_at = datetime.now(UTC).isoformat()
    timings = _Timings()
    with timings.stage("channels"):
        outcome = search(loaded, backend, config, focus)
    with timings.stage("graph"):
        expansion = expand_graph(
            loaded.graph, loaded.passages, outcome.candidates, focus, config.retrieval
        )
    with timings.stage("pool"):
        pool = build_pool(outcome.candidates, expansion, loaded.passages, config)
    with timings.stage("rerank"):
        ranked = rerank(
            pool.candidates,
            loaded.passages,
            reranker,
            focus,
            counters=counters,
            config=config,
        )
    with timings.stage("evidence"):
        selection = select_evidence(
            ranked,
            graph=loaded.graph,
            passages=loaded.passages,
            generation_id=loaded.generation_id,
            collection_version=loaded.collection_version,
            config=config,
        )
    used = {
        item.passage_id
        for item in selection.packet.items
        if item.passage_id is not None
    }
    candidates = tuple(
        Candidate.model_validate(
            {
                **candidate.model_dump(mode="json"),
                "selected": candidate.passage_id in used,
            },
            strict=True,
        )
        for candidate in ranked
    )
    return RetrievalRun(
        outcome=outcome,
        expansion=expansion,
        pool=pool,
        ranked=candidates,
        selection=selection,
        trace=_trace(
            _TraceParts(
                loaded=loaded,
                outcome=outcome,
                expansion=expansion,
                pool=pool,
                candidates=candidates,
                selection=selection,
                focus=focus,
                query_tokens=query_tokens,
                settings_fingerprint=_settings_fingerprint(config),
                started_at=started_at,
                stage_timings=timings.recorded(model_load_seconds),
            )
        ),
    )


class _Timings:
    """Collect monotonic stage durations for one run."""

    def __init__(self) -> None:
        """Start with no recorded stage."""
        self._recorded: list[StageTiming] = []

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Record the monotonic duration of one named stage."""
        started = time.monotonic()
        try:
            yield
        finally:
            self._recorded.append(
                StageTiming(stage=name, seconds=time.monotonic() - started)
            )

    def recorded(self, model_load_seconds: float) -> list[StageTiming]:
        """Return model-load time first, then the query stages."""
        return [
            StageTiming(stage="model_load", seconds=model_load_seconds),
            *self._recorded,
        ]


def _settings_fingerprint(config: AppConfig) -> str:
    """Return the identity of the query-side settings used by this run."""
    return fingerprint(
        {
            "embedding": config.embedding.model_dump(mode="json"),
            "keyword_index": config.keyword_index.model_dump(mode="json"),
            "retrieval": config.retrieval.model_dump(mode="json"),
            "reranker": config.reranker.model_dump(mode="json"),
        }
    )


@dataclass(frozen=True, slots=True)
class _TraceParts:
    """Stage results needed to build one validated retrieval trace."""

    loaded: LoadedGeneration
    outcome: SearchOutcome
    expansion: GraphExpansion
    pool: Pool
    candidates: tuple[Candidate, ...]
    selection: Selection
    focus: str
    query_tokens: int
    settings_fingerprint: str
    started_at: str
    stage_timings: list[StageTiming]


def _trace(parts: _TraceParts) -> RetrievalTrace:
    """Build the validated diagnostic trace for one run."""
    return RetrievalTrace(
        schema_version=1,
        generation_id=parts.loaded.generation_id,
        collection_version=parts.loaded.collection_version,
        settings_fingerprint=parts.settings_fingerprint,
        started_at=parts.started_at,
        focus=parts.focus,
        query_tokens=parts.query_tokens,
        stage_timings=parts.stage_timings,
        dense_hits=[
            ChannelHit(passage_id=passage_id, rank=rank, value=value)
            for rank, (passage_id, value) in enumerate(parts.outcome.dense_hits, 1)
        ],
        keyword_hits=[
            ChannelHit(passage_id=passage_id, rank=rank, value=value)
            for rank, (passage_id, value) in enumerate(parts.outcome.keyword_hits, 1)
        ],
        fused_candidates=list(parts.outcome.candidates),
        keyword_zero_score_exclusions=parts.outcome.keyword_zero_score_exclusions,
        graph_seeds=list(parts.expansion.seeds),
        graph_paths=list(parts.expansion.paths),
        direct_candidate_count=parts.pool.direct_count,
        direct_quota_before_dedup=parts.pool.direct_quota_before_dedup,
        graph_candidate_count=parts.pool.graph_count,
        graph_quota_before_dedup=parts.pool.graph_quota_before_dedup,
        candidates=list(parts.candidates),
        evidence_ids=[item.evidence_id for item in parts.selection.packet.items],
        exclusions=list(parts.selection.exclusions),
        limits_reached=list(parts.pool.limits_reached),
        missing_categories=list(parts.selection.missing_categories),
        provenance=list(parts.selection.provenance),
    )
