"""Index recovery: damaged bundles block reuse, and repair rebuilds from SQLite."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.contracts import ReviewRequest
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.ingest import BuildServices, run_ingestion
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.loader import LoadedGeneration, load_generation
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.services.session import WorkspaceSession
from fakes import FixedContextualizer, RecordingEmbedder

Damage = Callable[[WorkspaceSession, str], None]


def _bundle(session: WorkspaceSession, generation_id: str) -> Path:
    """Return the published bundle directory of one generation."""
    artifact = session.repository.artifact(generation_id)
    assert artifact is not None
    return session.paths.indexes.parent / artifact.directory


def _remove_dense(session: WorkspaceSession, generation_id: str) -> None:
    """Delete the dense index file."""
    (_bundle(session, generation_id) / "dense.usearch").unlink()


def _remove_keyword(session: WorkspaceSession, generation_id: str) -> None:
    """Delete the keyword index directory."""
    shutil.rmtree(_bundle(session, generation_id) / "keyword")


def _corrupt_dense(session: WorkspaceSession, generation_id: str) -> None:
    """Overwrite the dense index with bytes that cannot be loaded."""
    (_bundle(session, generation_id) / "dense.usearch").write_bytes(b"not an index")


def _remove_bundle(session: WorkspaceSession, generation_id: str) -> None:
    """Delete the whole published bundle directory."""
    shutil.rmtree(_bundle(session, generation_id))


DAMAGE = [
    pytest.param(_remove_dense, id="missing-dense"),
    pytest.param(_remove_keyword, id="missing-keyword"),
    pytest.param(_corrupt_dense, id="corrupt-dense"),
    pytest.param(_remove_bundle, id="missing-bundle"),
]


def _reload(session: WorkspaceSession, generation_id: str) -> LoadedGeneration:
    """Load and verify one generation through the normal retrieval path."""
    return load_generation(
        session.repository,
        passages=session.passages,
        indexes_dir=session.paths.indexes,
        config=session.config,
        generation_id=generation_id,
    )


def _services(
    session: WorkspaceSession,
    counters: TokenCounters,
    embedder: RecordingEmbedder,
    contextualizer: FixedContextualizer,
) -> BuildServices:
    """Return direct build collaborators for one fixture session."""
    return BuildServices(
        config=session.config,
        repository=session.repository,
        passages=session.passages,
        indexes_dir=session.paths.indexes,
        counters=counters,
        backend=embedder,
        contextualizer=contextualizer,
        progress=None,
    )


class TestReuseValidation:
    """A damaged bundle is never accepted as reusable."""

    @pytest.mark.parametrize("damage", DAMAGE)
    def test_damaged_bundle_requires_repair_instead_of_reuse(
        self,
        fake_session: WorkspaceSession,
        loaded_generation: LoadedGeneration,
        source_path: Path,
        damage: Damage,
    ) -> None:
        generation_id = loaded_generation.generation_id
        damage(fake_session, generation_id)
        prepared = fake_session.prepare_ingestion(source_path)
        assert prepared.plan.reuse_generation is False
        assert prepared.plan.repair_required is True
        assert prepared.plan.uncached_context_count == 0
        with pytest.raises(AtlasReviewError) as caught:
            fake_session.build_generation(prepared)
        assert caught.value.code == "index_repair_required"
        assert fake_session.repository.active_generation_id() == generation_id

    def test_plan_reports_reuse_only_for_a_loadable_bundle(
        self,
        fake_session: WorkspaceSession,
        loaded_generation: LoadedGeneration,
        source_path: Path,
    ) -> None:
        prepared = fake_session.prepare_ingestion(source_path)
        assert prepared.plan.reuse_generation is True
        assert prepared.plan.repair_required is False
        _remove_dense(fake_session, loaded_generation.generation_id)
        assert (
            fake_session.prepare_ingestion(source_path).plan.reuse_generation is False
        )


class TestRepair:
    """Repair rebuilds both indexes from stored rows without any model call."""

    @pytest.mark.parametrize("damage", DAMAGE)
    def test_repair_restores_a_loadable_bundle(
        self,
        fake_session: WorkspaceSession,
        loaded_generation: LoadedGeneration,
        contextualizer: FixedContextualizer,
        embedder: RecordingEmbedder,
        damage: Damage,
    ) -> None:
        generation_id = loaded_generation.generation_id
        artifact_before = fake_session.repository.artifact(generation_id)
        requests_before = len(contextualizer.calls)
        embeddings_before = len(embedder.inputs)
        old_bundle = _bundle(fake_session, generation_id)
        damage(fake_session, generation_id)

        outcome = fake_session.rebuild_indexes(generation_id)

        assert outcome.generation_id == generation_id
        assert outcome.passage_count == len(loaded_generation.passage_ids)
        assert outcome.dense_count == len(loaded_generation.passage_ids)
        assert outcome.keyword_count == len(loaded_generation.passage_ids)
        assert len(contextualizer.calls) == requests_before
        assert len(embedder.inputs) == embeddings_before
        assert not old_bundle.exists()
        artifact = fake_session.repository.artifact(generation_id)
        assert artifact is not None
        assert artifact_before is not None
        assert artifact.directory != artifact_before.directory
        manifest = fake_session.repository.manifest(generation_id)
        assert manifest.artifact_manifest_sha256 == artifact.manifest_sha256
        reloaded = _reload(fake_session, generation_id)
        assert reloaded.passage_ids == loaded_generation.passage_ids
        assert set(reloaded.passages) == set(loaded_generation.passages)
        assert fake_session.repository.active_generation_id() == generation_id

    def test_referenced_artifact_directory_is_never_removed(
        self,
        fake_session: WorkspaceSession,
        loaded_generation: LoadedGeneration,
    ) -> None:
        artifact = fake_session.repository.artifact(loaded_generation.generation_id)
        assert artifact is not None
        assert fake_session.repository.artifact_is_referenced(artifact.directory)
        assert not fake_session.repository.artifact_is_referenced("indexes/absent")

    def test_repair_uses_stored_vectors_and_retrieval_text(
        self,
        fake_session: WorkspaceSession,
        loaded_generation: LoadedGeneration,
    ) -> None:
        generation_id = loaded_generation.generation_id
        stored = fake_session.passages.passages(generation_id)
        _remove_dense(fake_session, generation_id)

        fake_session.rebuild_indexes(generation_id)

        reloaded = _reload(fake_session, generation_id)
        texts = [reloaded.passages[key].retrieval_text for key in reloaded.passage_ids]
        assert texts == [passage.retrieval_text for passage in stored]
        rows = fake_session.passages.index_rows(generation_id)
        assert [row.passage_id for row in rows] == list(reloaded.passage_ids)

    def test_repair_refuses_changed_analysis_settings(
        self,
        fake_session: WorkspaceSession,
        loaded_generation: LoadedGeneration,
    ) -> None:
        fake_session.save_config(
            AppConfig.model_validate(
                {
                    **fake_session.config.model_dump(mode="python"),
                    "keyword_index": {
                        **fake_session.config.keyword_index.model_dump(mode="python"),
                        "b": 0.5,
                    },
                },
                strict=False,
            )
        )
        _remove_dense(fake_session, loaded_generation.generation_id)
        with pytest.raises(AtlasReviewError) as caught:
            fake_session.rebuild_indexes(loaded_generation.generation_id)
        assert caught.value.code == "query_configuration_mismatch"

    def test_unknown_generation_is_not_repairable(
        self, fake_session: WorkspaceSession
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            fake_session.rebuild_indexes("00000000-0000-0000-0000-000000000000")
        assert caught.value.code == "generation_not_repairable"


class TestReviewReferences:
    """Repair keeps the generation and passage identities reviews refer to."""

    def test_saved_review_still_resolves_after_repair(
        self,
        review_session: WorkspaceSession,
        review_brief: ReviewRequest,
    ) -> None:
        generation_id = review_session.require_ready_generation_id()
        draft = review_session.retrieve_review(review_brief)
        stored = review_session.open_review(draft.review_id)
        _remove_dense(review_session, generation_id)

        review_session.rebuild_indexes(generation_id)

        reopened = review_session.open_review(draft.review_id)
        assert reopened.generation_id == generation_id
        assert reopened.evidence == stored.evidence
        assert reopened.request == stored.request
        again = review_session.retrieve_review(review_brief)
        assert again.run.trace.generation_id == generation_id


class TestDirectBuild:
    """The service-level build also rejects a damaged ready generation."""

    def test_run_ingestion_raises_repair_required_for_damaged_bundle(
        self,
        fake_session: WorkspaceSession,
        corpus: ParsedCorpus,
        counters: TokenCounters,
        embedder: RecordingEmbedder,
        contextualizer: FixedContextualizer,
    ) -> None:
        services = _services(fake_session, counters, embedder, contextualizer)
        first = run_ingestion(corpus, source_bytes=b"fixture", services=services)
        _remove_dense(fake_session, first.generation_id)
        calls = len(contextualizer.calls)
        with pytest.raises(AtlasReviewError) as caught:
            run_ingestion(corpus, source_bytes=b"fixture", services=services)
        assert caught.value.code == "index_repair_required"
        assert len(contextualizer.calls) == calls
