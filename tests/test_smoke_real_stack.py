"""Real-library smoke test: models, indexes, storage, and search.

Run explicitly, because it downloads model artifacts and needs sockets::

    ATLAS_REVIEW_SMOKE=1 uv run pytest -m smoke --force-enable-socket

Provider calls are replaced by a fixed contextualizer so the test needs no
credentials; FastEmbed, bm25s, USearch, SQLite, and msgspec are all real.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from atlas_review.config import EMBEDDING_DIMENSION
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.ingest import BuildServices, run_ingestion
from atlas_review.retrieval.embeddings import FastEmbedBackend, query_vector
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
    finally:
        session.close()
