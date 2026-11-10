"""Contextualized ingestion integration: storage, cache reuse, and atomicity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from atlas_review.config import AppConfig, LLMConfig
from atlas_review.contracts import ChunkingIdentity
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.chunk import build_source_chunks
from atlas_review.corpus.ingest import (
    BuildServices,
    IngestionOutcome,
    IngestionPlan,
    plan_ingestion,
    run_ingestion,
)
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.services import session as session_module
from atlas_review.services.session import WorkspaceSession
from fakes import FixedContextualizer, RecordingEmbedder


@dataclass(frozen=True, slots=True)
class Builder:
    """Run explicit builds against one prepared workspace.

    Instance Attributes:
        session (WorkspaceSession): Open workspace session.
        corpus (ParsedCorpus): Parsed fixture records.
        counters (TokenCounters): Retrieval tokenizers.
        embedder (RecordingEmbedder): Recording embedding backend.
    """

    session: WorkspaceSession
    corpus: ParsedCorpus
    counters: TokenCounters
    embedder: RecordingEmbedder

    def build(
        self,
        contextualizer: FixedContextualizer | None,
        *,
        config: AppConfig | None = None,
        progress: list[str] | None = None,
    ) -> IngestionOutcome:
        """Run one explicit build with the given contextualizer."""
        return run_ingestion(
            self.corpus,
            source_bytes=b"fixture",
            services=BuildServices(
                config=config or self.session.config,
                repository=self.session.repository,
                passages=self.session.passages,
                indexes_dir=self.session.paths.indexes,
                counters=self.counters,
                backend=self.embedder,
                contextualizer=contextualizer,
                progress=progress.append if progress is not None else None,
            ),
        )

    def plan(self, *, config: AppConfig | None = None) -> IngestionPlan:
        """Compute the LLM-free plan for the same inputs."""
        return plan_ingestion(
            self.corpus,
            config=config or self.session.config,
            repository=self.session.repository,
            passages=self.session.passages,
            indexes_dir=self.session.paths.indexes,
            counters=self.counters,
        )


@pytest.fixture
def builder(
    session: WorkspaceSession,
    corpus: ParsedCorpus,
    counters: TokenCounters,
    embedder: RecordingEmbedder,
) -> Builder:
    """Return a builder bound to the session and the deterministic fakes."""
    return Builder(session, corpus, counters, embedder)


def changed_config(config: AppConfig, section: str, **changes: object) -> AppConfig:
    """Return a validated configuration with one section patched."""
    return AppConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            section: {**getattr(config, section).model_dump(mode="python"), **changes},
        },
        strict=False,
    )


class TestBuild:
    """One build stores passages and publishes one joint index bundle."""

    def test_build_stores_passages_and_activates_the_generation(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        progress: list[str] = []
        outcome = builder.build(contextualizer, progress=progress)
        assert (
            outcome.generation_id == builder.session.repository.active_generation_id()
        )
        assert outcome.passage_count > 0
        assert outcome.requested_context_count == outcome.context_count
        assert outcome.dense_count == outcome.passage_count
        assert outcome.keyword_count == outcome.passage_count
        assert len(contextualizer.calls) == outcome.context_count
        assert progress[0]
        assert progress[-1].startswith("activated generation")

    def test_stored_passages_cover_the_planned_chunks(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        outcome = builder.build(contextualizer)
        stored = builder.session.passages.passages(outcome.generation_id)
        chunks = build_source_chunks(
            builder.corpus.entities,
            builder.corpus.edges,
            builder.counters.embedding,
            builder.session.config.chunking,
        )
        assert len(stored) == len(chunks)
        assert [passage.bm25_ordinal for passage in stored] == list(range(len(stored)))
        for passage in stored:
            assert passage.retrieval_text.startswith(passage.header)
            assert passage.context_text in passage.retrieval_text
            assert passage.source_text in passage.retrieval_text
            assert len(passage.source_text) == passage.source_end - passage.source_start

    def test_generation_row_records_the_artifact_and_counts(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        outcome = builder.build(contextualizer)
        manifest = builder.session.repository.manifest(outcome.generation_id)
        artifact = builder.session.repository.artifact(outcome.generation_id)
        assert artifact is not None
        assert manifest.artifact_manifest_sha256 == artifact.manifest_sha256
        assert manifest.passage_count == outcome.passage_count
        assert manifest.context_count == outcome.context_count
        assert manifest.entity_count == len(builder.corpus.entities)
        assert manifest.settings_fingerprint is not None
        assert isinstance(manifest.chunking, ChunkingIdentity)
        assert manifest.embedding is not None
        assert manifest.lexical is not None
        assert manifest.contextualization is not None
        assert (
            manifest.contextualization.reasoning_effort
            == builder.session.config.llm.reasoning_effort
        )
        bundle = builder.session.paths.indexes.parent / artifact.directory
        assert (bundle / "artifacts.json").is_file()
        assert (bundle / "dense.usearch").is_file()
        assert (bundle / "passage_ids.msgpack").is_file()
        assert (bundle / "keyword" / "params.index.json").is_file()

    def test_build_writes_vectors_to_passage_rows(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        outcome = builder.build(contextualizer)
        rows = (
            builder.session.database.connect()
            .execute(
                "SELECT vector_f32_le FROM passages WHERE generation_id = ?",
                (outcome.generation_id,),
            )
            .fetchall()
        )
        assert len(rows) == outcome.passage_count
        assert all(len(row["vector_f32_le"]) == 768 * 4 for row in rows)


class TestContextCache:
    """Accepted context is cached once and reused by every later build."""

    def test_cached_context_is_reused_without_new_requests(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        first = builder.build(contextualizer)
        assert len(contextualizer.calls) == first.context_count
        second = builder.build(
            contextualizer,
            config=changed_config(
                builder.session.config, "vector_index", connectivity=32
            ),
        )
        assert second.generation_id != first.generation_id
        assert second.requested_context_count == 0
        assert second.cached_context_count == second.context_count
        assert len(contextualizer.calls) == first.context_count

    def test_plan_reports_cache_reuse_before_building(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        first = builder.plan()
        assert first.cached_context_count == 0
        assert first.uncached_context_count == first.chunk_count
        assert first.reuse_generation is False
        assert first.endpoint_host == "your-provider.example"
        builder.build(contextualizer)
        again = builder.plan()
        assert again.reuse_generation is True
        assert again.generation_id == first.generation_id

    def test_context_entries_are_stored_with_document_and_chunk_hashes(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        outcome = builder.build(contextualizer)
        rows = (
            builder.session.database.connect()
            .execute("SELECT * FROM context_cache")
            .fetchall()
        )
        assert len(rows) == outcome.context_count
        for row in rows:
            assert len(row["document_sha256"]) == 64
            assert len(row["chunk_sha256"]) == 64
            assert row["contextualizer_fingerprint"] == row["cache_key"]
            assert row["context_text"].strip()

    def test_regeneration_revision_forces_new_requests(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        first = builder.build(contextualizer)
        calls = len(contextualizer.calls)
        second = builder.build(
            contextualizer,
            config=changed_config(
                builder.session.config, "contextualization", regeneration_revision=1
            ),
        )
        assert second.generation_id != first.generation_id
        assert second.requested_context_count == second.context_count
        assert len(contextualizer.calls) == calls + second.context_count


class TestGenerationIdentity:
    """Every contextualization input shapes the generation identity."""

    @pytest.mark.parametrize("effort", ["none", "low", "high"])
    def test_changed_reasoning_effort_changes_the_generation_id(
        self, builder: Builder, effort: str
    ) -> None:
        default = builder.plan()
        changed = builder.plan(
            config=changed_config(
                builder.session.config, "llm", reasoning_effort=effort
            )
        )
        assert changed.settings_fingerprint != default.settings_fingerprint
        assert changed.generation_id != default.generation_id


class TestReuse:
    """An identical ready generation is activated instead of rebuilt."""

    def test_identical_build_reactivates_the_ready_generation(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        first = builder.build(contextualizer)
        calls = len(contextualizer.calls)
        second = builder.build(contextualizer)
        assert second.generation_id == first.generation_id
        assert second.reused_generation is True
        assert second.requested_context_count == 0
        assert len(contextualizer.calls) == calls
        assert len(builder.session.generations()) == 1


class TestFailureIsolation:
    """A failed build leaves the previously active generation active."""

    def test_missing_contextualizer_fails_without_storing_a_generation(
        self, builder: Builder
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            builder.build(None)
        assert caught.value.code == "contextualizer_required"
        assert builder.session.repository.active_generation_id() is None
        assert builder.session.generations() == ()

    def test_invalid_response_keeps_the_previous_generation_active(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        first = builder.build(contextualizer)
        broken = FixedContextualizer(reply_builder=lambda _: "not json")
        with pytest.raises(AtlasReviewError) as caught:
            builder.build(
                broken,
                config=changed_config(
                    builder.session.config, "contextualization", regeneration_revision=1
                ),
            )
        assert caught.value.code == "invalid_context_response"
        assert builder.session.repository.active_generation_id() == first.generation_id
        assert [summary.status for summary in builder.session.generations()] == [
            "ready"
        ]

    def test_failed_publication_stores_no_generation(
        self,
        builder: Builder,
        contextualizer: FixedContextualizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def explode(*_: object, **__: object) -> None:
            raise AtlasReviewError(
                code="bundle_write_failed",
                stage="index_integrity",
                message="simulated bundle failure",
            )

        monkeypatch.setattr("atlas_review.corpus.ingest.build_bundle", explode)
        with pytest.raises(AtlasReviewError):
            builder.build(contextualizer)
        assert builder.session.generations() == ()
        assert builder.session.repository.active_generation_id() is None


class TestQueryText:
    """Passages are embedded unchanged; only searches embed a query."""

    def test_passage_batches_carry_no_query_instruction(
        self, builder: Builder, contextualizer: FixedContextualizer
    ) -> None:
        builder.build(contextualizer)
        assert builder.embedder.inputs
        prefix = builder.session.config.embedding.query_prefix
        assert all(
            not text.startswith(prefix)
            for batch in builder.embedder.inputs
            for text in batch
        )
        assert all(len(batch) > 1 for batch in builder.embedder.inputs)


def test_build_generation_passes_the_configured_retries_and_reporter(
    monkeypatch: pytest.MonkeyPatch,
    fake_session: WorkspaceSession,
    source_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def factory(
        _config: LLMConfig,
        *,
        automatic_retries: int = 0,
        notify: Callable[[str], None] | None = None,
    ) -> FixedContextualizer:
        captured.update(retries=automatic_retries, notify=notify)
        return FixedContextualizer()

    fake_session.save_config(
        changed_config(fake_session.config, "contextualization", automatic_retries=3)
    )
    monkeypatch.setattr(session_module, "build_contextualizer", factory)
    prepared = fake_session.prepare_ingestion(source_path)
    progress: list[str] = []
    fake_session.build_generation(prepared, progress=progress.append)
    assert captured["retries"] == 3
    assert captured["notify"] == progress.append
