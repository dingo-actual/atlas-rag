"""End-to-end retrieval: fusion, graph expansion, reranking, and trace."""

from __future__ import annotations

import pytest

from atlas_review.config import AppConfig, RetrievalConfig
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.embeddings import EmbeddingBackend
from atlas_review.retrieval.loader import LoadedGeneration
from atlas_review.retrieval.pipeline import RetrievalRun, retrieve
from atlas_review.retrieval.tokenizers import TokenCounters
from fakes import RecordingReranker, WhitespaceTokenCounter

FOCUS = "indirect prompt injection through retrieved content"
CASE = "AML.CS0035"
TECHNIQUE = "AML.T0051.001"
MITIGATION = "AML.M0033"
COUNTERS = TokenCounters(
    embedding=WhitespaceTokenCounter(), reranker=WhitespaceTokenCounter()
)


def _config(**change: int) -> AppConfig:
    """Return the default configuration with explicit retrieval overrides."""
    return AppConfig.model_validate(
        {
            **AppConfig().model_dump(mode="python"),
            "retrieval": {
                **RetrievalConfig().model_dump(mode="python"),
                **change,
            },
        },
        strict=False,
    )


def _retrieve(
    loaded_generation: LoadedGeneration,
    embedder: EmbeddingBackend,
    reranker: RecordingReranker,
    config: AppConfig,
    focus: str = FOCUS,
) -> RetrievalRun:
    """Run the retrieval pipeline over the built fixture."""
    return retrieve(loaded_generation, embedder, reranker, config, focus, COUNTERS)


NARROW = {"dense_k": 1, "bm25_k": 1}


class TestRetrievalStages:
    """Every stage contributes to one traceable run."""

    def test_run_reports_both_channels_and_fused_candidates(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(loaded_generation, embedder, reranker, AppConfig())
        assert run.outcome.dense_hits
        assert run.outcome.keyword_hits
        assert run.outcome.candidates
        assert run.selection.packet.items

    def test_trace_records_channel_lists_ranks_and_provenance(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(loaded_generation, embedder, reranker, AppConfig())
        trace = run.trace
        assert [hit.passage_id for hit in trace.dense_hits] == [
            passage_id for passage_id, _ in run.outcome.dense_hits
        ]
        assert [hit.rank for hit in trace.dense_hits] == list(
            range(1, len(trace.dense_hits) + 1)
        )
        assert [hit.passage_id for hit in trace.keyword_hits] == [
            passage_id for passage_id, _ in run.outcome.keyword_hits
        ]
        assert trace.fused_candidates == list(run.outcome.candidates)
        assert trace.query_tokens > 0
        assert trace.generation_id == loaded_generation.generation_id
        assert trace.collection_version == loaded_generation.collection_version
        assert trace.evidence_ids == [
            item.evidence_id for item in run.selection.packet.items
        ]
        assert trace.provenance

    def test_reranked_pool_is_ordered_by_score_then_passage(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(loaded_generation, embedder, reranker, AppConfig())
        scores = [candidate.reranker_score or 0.0 for candidate in run.ranked]
        assert all(candidate.reranker_score is not None for candidate in run.ranked)
        assert scores == sorted(scores, reverse=True)
        for previous, following in zip(run.ranked, run.ranked[1:], strict=False):
            if previous.reranker_score == following.reranker_score:
                assert previous.passage_id < following.passage_id
        assert reranker.calls[0][0] == FOCUS
        assert all(FOCUS not in document for document in reranker.calls[0][1])

    def test_graph_expansion_adds_a_candidate_absent_from_both_channels(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(loaded_generation, embedder, reranker, _config(**NARROW))
        initial = {
            passage_id
            for passage_id, _ in (*run.outcome.dense_hits, *run.outcome.keyword_hits)
        }
        graph_only = [
            candidate
            for candidate in run.trace.candidates
            if "graph" in candidate.channels
            and "dense" not in candidate.channels
            and "bm25" not in candidate.channels
        ]
        assert graph_only
        assert any(candidate.passage_id not in initial for candidate in graph_only)
        passages = {candidate.passage_id: candidate for candidate in graph_only}
        assert any(
            candidate.graph_paths and candidate.graph_paths[0].hops
            for candidate in passages.values()
        )

    def test_pool_is_capped_and_keeps_direct_capacity(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(
            loaded_generation,
            embedder,
            reranker,
            _config(**{"direct_candidate_limit": 4, "rerank_candidate_limit": 6}),
        )
        assert len(run.pool.candidates) <= 6
        assert run.pool.direct_count <= 4
        assert len(run.ranked) == len(run.pool.candidates)

    def test_run_is_repeatable(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        first = _retrieve(loaded_generation, embedder, reranker, _config(**NARROW))
        second = _retrieve(loaded_generation, embedder, reranker, _config(**NARROW))
        assert [candidate.passage_id for candidate in first.ranked] == [
            candidate.passage_id for candidate in second.ranked
        ]
        assert first.trace.graph_paths == second.trace.graph_paths
        assert first.trace.evidence_ids == second.trace.evidence_ids

    def test_exact_identifier_is_an_anchor_before_fused_seeds(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(
            loaded_generation,
            embedder,
            reranker,
            AppConfig(),
            focus=f"compare {CASE} with retrieved content",
        )
        assert run.trace.graph_seeds[0].entity_id == CASE
        assert run.trace.graph_seeds[0].origin == "exact_id"

    def test_unknown_explicit_identifier_fails_before_reranking(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            _retrieve(
                loaded_generation,
                embedder,
                reranker,
                AppConfig(),
                focus="look at AML.T0999",
            )
        assert caught.value.code == "unknown_explicit_id"
        assert caught.value.stage == "retrieval"
        assert reranker.calls == []

    def test_technique_seed_reaches_a_connected_mitigation(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(
            loaded_generation,
            embedder,
            reranker,
            _config(exact_id_limit=1, seed_entity_limit=1),
            focus=f"{TECHNIQUE} mitigation review",
        )
        assert [seed.entity_id for seed in run.trace.graph_seeds] == [TECHNIQUE]
        assert any(path.entity_id == MITIGATION for path in run.trace.graph_paths)

    def test_trace_records_identities_timings_and_quota_counts(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(
            loaded_generation,
            embedder,
            reranker,
            _config(**{"direct_candidate_limit": 4, "rerank_candidate_limit": 6}),
        )
        trace = run.trace
        assert trace.started_at
        assert len(trace.settings_fingerprint) == 64
        stages = [timing.stage for timing in trace.stage_timings]
        assert stages == [
            "model_load",
            "channels",
            "graph",
            "pool",
            "rerank",
            "evidence",
        ]
        assert all(timing.seconds >= 0.0 for timing in trace.stage_timings)
        assert trace.direct_candidate_count <= trace.direct_quota_before_dedup
        assert trace.graph_candidate_count <= trace.graph_quota_before_dedup
        assert trace.graph_quota_before_dedup == len(run.expansion.candidates)
        assert trace.keyword_zero_score_exclusions >= 0

    def test_trace_records_paths_when_every_result_is_already_a_seed(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(loaded_generation, embedder, reranker, AppConfig())
        traced = {
            (path.seed_entity_id, path.entity_id) for path in run.trace.graph_paths
        }
        assert any(
            path.graph_paths for path in run.trace.candidates if path.graph_paths
        )
        for candidate in run.trace.candidates:
            for path in candidate.graph_paths:
                assert (path.seed_entity_id, path.entity_id) in traced

    def test_trace_keeps_at_most_two_paths_per_candidate(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(loaded_generation, embedder, reranker, AppConfig())
        assert all(
            len(candidate.graph_paths) <= 2 for candidate in run.trace.candidates
        )

    def test_fused_candidate_keeps_the_pool_graph_path(
        self,
        loaded_generation: LoadedGeneration,
        embedder: EmbeddingBackend,
        reranker: RecordingReranker,
    ) -> None:
        run = _retrieve(
            loaded_generation,
            embedder,
            reranker,
            _config(dense_k=1, bm25_k=1),
            focus=f"{CASE} review",
        )
        channels = {
            candidate.passage_id: set(candidate.channels)
            for candidate in run.outcome.candidates
        }
        overlap = [
            candidate
            for candidate in run.trace.candidates
            if candidate.graph_paths and channels.get(candidate.passage_id)
        ]
        assert overlap
        assert all(candidate.channels[0] in {"dense", "bm25"} for candidate in overlap)
