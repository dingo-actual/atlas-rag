"""Real-library smoke test: models, indexes, graph, reranker, and evidence.

Run explicitly, because it downloads model artifacts and needs sockets::

    ATLAS_REVIEW_SMOKE=1 uv run pytest -m smoke --force-enable-socket

Provider calls are replaced by a fixed contextualizer so the test needs no
credentials; FastEmbed embeddings and cross-encoder, rustworkx, bm25s, USearch,
SQLite, and msgspec are all real.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from atlas_review.config import EMBEDDING_DIMENSION, AppConfig
from atlas_review.contracts import Passage, ReviewRequest
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.ingest import BuildServices, run_ingestion
from atlas_review.retrieval.embeddings import FastEmbedBackend, query_vector
from atlas_review.retrieval.hybrid import check_focus
from atlas_review.retrieval.loader import LoadedGeneration, load_generation
from atlas_review.retrieval.pipeline import retrieve
from atlas_review.retrieval.rerank import FastEmbedReranker
from atlas_review.services.session import WorkspaceSession
from fakes import FixedContextualizer

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.environ.get("ATLAS_REVIEW_SMOKE") != "1",
        reason="set ATLAS_REVIEW_SMOKE=1 to run the real-library smoke test",
    ),
]

FOCUS = "indirect prompt injection through retrieved content"


def test_real_stack_builds_and_searches(
    workspace_root: Path, corpus: ParsedCorpus, source_path: Path
) -> None:
    session = WorkspaceSession(workspace_root)
    counters = session.token_counters()
    backend = FastEmbedBackend(session.config.embedding, session.paths.models)
    try:
        outcome = run_ingestion(
            corpus,
            source_bytes=source_path.read_bytes(),
            services=BuildServices(
                config=session.config,
                repository=session.repository,
                passages=session.passages,
                indexes_dir=session.paths.indexes,
                counters=counters,
                backend=backend,
                contextualizer=FixedContextualizer(),
                progress=None,
            ),
        )
        assert outcome.passage_count > 0
        stored = session.passages.passages(outcome.generation_id)
        assert max(passage.reranker_tokens for passage in stored) <= (
            session.config.chunking.passage_max_tokens
        )

        vector = query_vector(backend, session.config.embedding, FOCUS)
        assert vector.shape == (EMBEDDING_DIMENSION,)
        assert np.isfinite(vector).all()
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
        assert counters.embedding.name.startswith(session.config.embedding.model_name)
        assert counters.reranker.name.startswith(session.config.reranker.model_name)

        found = session.search(FOCUS)
        assert found.dense_hits
        assert found.keyword_hits
        assert found.candidates
        assert all(passage.retrieval_text for passage in found.passages)

        unseen = session.search("zzzz qqqq")
        assert unseen.keyword_hits == ()
        assert unseen.dense_hits

        run = session.retrieve(FOCUS)
        assert run.trace.query_tokens > 0
        assert run.trace.graph_seeds
        assert run.trace.graph_paths
        assert run.trace.stage_timings
        assert run.trace.stage_timings[0].stage == "model_load"
        assert run.trace.stage_timings[0].seconds > 0.0
        assert len(run.trace.settings_fingerprint) == 64
        assert all(candidate.reranker_score is not None for candidate in run.ranked)
        assert run.selection.packet.items
        assert run.selection.provenance
        used = {
            item.passage_id
            for item in run.selection.packet.items
            if item.passage_id is not None
        }
        assert used <= {passage.passage_id for passage in stored}
        assert all(
            stored_passage.context_text not in item.text
            for item in run.selection.packet.items
            for stored_passage in stored
        )

        narrow = _narrow(session.config)
        loaded = _repaired_bundle(session, outcome.generation_id, stored)
        single = retrieve(
            loaded,
            backend,
            FastEmbedReranker(session.config.reranker, session.paths.models),
            narrow,
            FOCUS,
            counters,
        )
        assert any(seed.channels == ["bm25"] for seed in single.trace.graph_seeds)
        initial = {
            passage_id
            for passage_id, _ in (
                *single.outcome.dense_hits,
                *single.outcome.keyword_hits,
            )
        }
        assert any(
            candidate.passage_id not in initial for candidate in single.trace.candidates
        )
        assert all(
            len(candidate.graph_paths) <= 2 for candidate in single.trace.candidates
        )

        examples = Path(__file__).resolve().parents[1] / "examples"
        for path in sorted(examples.glob("*.json")):
            brief = ReviewRequest.model_validate_json(
                path.read_text(encoding="utf-8"), strict=True
            )
            check_focus(brief.retrieval_focus, counters, session.config)
    finally:
        session.close()


def _repaired_bundle(
    session: WorkspaceSession, generation_id: str, stored: Sequence[Passage]
) -> LoadedGeneration:
    """Rebuild both indexes from SQLite, then load the repaired bundle."""
    repaired = session.rebuild_indexes(generation_id)
    assert repaired.passage_count == len(stored)
    assert repaired.dense_count == len(stored)
    assert repaired.keyword_count == len(stored)
    loaded = load_generation(
        session.repository,
        passages=session.passages,
        indexes_dir=session.paths.indexes,
        config=session.config,
        generation_id=generation_id,
    )
    assert loaded.passage_ids == tuple(passage.passage_id for passage in stored)
    return loaded


def _narrow(config: AppConfig) -> AppConfig:
    """Return the same configuration with one candidate per initial channel."""
    return AppConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "retrieval": {
                **config.retrieval.model_dump(mode="python"),
                "dense_k": 1,
                "bm25_k": 1,
            },
        },
        strict=False,
    )
