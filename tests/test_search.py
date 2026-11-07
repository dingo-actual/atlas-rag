"""Retrieval integration: both channels, fusion, and index integrity."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.corpus.atlas_v6 import ParsedCorpus
from atlas_review.corpus.ingest import BuildServices, run_ingestion
from atlas_review.errors import AtlasReviewError
from atlas_review.retrieval.hybrid import SearchOutcome, check_focus, search
from atlas_review.retrieval.loader import LoadedGeneration, load_generation
from atlas_review.retrieval.tokenizers import TokenCounters
from atlas_review.services.session import WorkspaceSession
from fakes import FixedContextualizer, RecordingEmbedder


@pytest.fixture
def session(
    workspace_root: Path, counters: TokenCounters, embedder: RecordingEmbedder
) -> Iterator[WorkspaceSession]:
    """Return a session bound to the deterministic fakes."""
    opened = WorkspaceSession(workspace_root, counters=counters, embedding=embedder)
    try:
        yield opened
    finally:
        opened.close()


@pytest.fixture
def built(
    session: WorkspaceSession,
    corpus: ParsedCorpus,
    counters: TokenCounters,
    embedder: RecordingEmbedder,
    contextualizer: FixedContextualizer,
) -> SearchOutcome:
    """Build the fixture generation once and return one retrieval run."""
    run_ingestion(
        corpus,
        source_bytes=b"fixture",
        services=BuildServices(
            config=session.config,
            repository=session.repository,
            passages=session.passages,
            indexes_dir=session.paths.indexes,
            counters=counters,
            backend=embedder,
            contextualizer=contextualizer,
            progress=None,
        ),
    )
    return session.search("indirect prompt injection through retrieved content")


def _loaded(session: WorkspaceSession) -> LoadedGeneration:
    """Return the verified bundle of the active generation."""
    return load_generation(
        session.repository,
        passages=session.passages,
        indexes_dir=session.paths.indexes,
        config=session.config,
        generation_id=session.require_ready_generation_id(),
    )


class TestSearch:
    """Both channels cover the same passages and fuse into ranked candidates."""

    def test_search_reports_both_channel_hits(self, built: SearchOutcome) -> None:
        assert built.dense_hits
        assert built.keyword_hits
        assert built.candidates
        assert built.collection_version

    def test_candidates_are_ordered_by_fusion_score(self, built: SearchOutcome) -> None:
        scores = [candidate.fusion_score for candidate in built.candidates]
        assert scores == sorted(scores, reverse=True)
        assert len(built.passages) == len(built.candidates)

    def test_candidates_keep_channel_provenance(self, built: SearchOutcome) -> None:
        fused = [
            candidate
            for candidate in built.candidates
            if set(candidate.channels) == {"dense", "bm25"}
        ]
        assert fused
        for candidate in fused:
            assert candidate.dense_rank is not None
            assert candidate.bm25_rank is not None
            assert candidate.dense_distance is not None
            assert candidate.bm25_score is not None

    def test_keyword_channel_ranks_the_matching_case_study(
        self, session: WorkspaceSession, built: SearchOutcome
    ) -> None:
        del built
        outcome = session.search("AML.CS0035 Slack AI indirect prompt injection")
        owners = {
            passage.entity_id
            for passage in outcome.passages
            if passage.owner_kind == "case_study"
        }
        assert "AML.CS0035" in owners
        assert outcome.keyword_hits

    def test_query_is_embedded_with_the_instruction_once(
        self,
        built: SearchOutcome,
        embedder: RecordingEmbedder,
        session: WorkspaceSession,
    ) -> None:
        del built
        prefix = session.config.embedding.query_prefix
        queries = embedder.query_inputs
        assert queries == [
            f"{prefix}indirect prompt injection through retrieved content"
        ]

    def test_every_hit_resolves_to_a_stored_passage(
        self, built: SearchOutcome, session: WorkspaceSession
    ) -> None:
        stored = {
            passage.passage_id
            for passage in session.passages.passages(built.generation_id)
        }
        assert {passage_id for passage_id, _ in built.dense_hits} <= stored
        assert {passage_id for passage_id, _ in built.keyword_hits} <= stored


class TestFocusChecks:
    """Blank and oversized focuses are rejected before any channel runs."""

    def test_blank_focus_is_rejected(
        self, session: WorkspaceSession, counters: TokenCounters
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            check_focus("   ", counters, session.config)
        assert caught.value.code == "blank_focus"

    def test_overlong_focus_is_rejected(
        self, session: WorkspaceSession, counters: TokenCounters
    ) -> None:
        focus = " ".join(["word"] * (session.config.retrieval.query_max_tokens + 1))
        with pytest.raises(AtlasReviewError) as caught:
            check_focus(focus, counters, session.config)
        assert caught.value.code == "query_too_long"

    def test_search_requires_an_active_generation(
        self, session: WorkspaceSession
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            session.search("prompt injection")
        assert caught.value.code == "no_active_generation"


class TestBundleIntegrity:
    """Tampered, missing, or incompatible bundles are refused before search."""

    def test_tampered_manifest_is_rejected(
        self, built: SearchOutcome, session: WorkspaceSession
    ) -> None:
        artifact = session.repository.artifact(built.generation_id)
        assert artifact is not None
        manifest_path = (
            session.paths.indexes.parent / artifact.directory / "artifacts.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["passage_count"] = manifest["passage_count"] + 1
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(AtlasReviewError) as caught:
            session.search("prompt injection")
        assert caught.value.code == "artifact_manifest_mismatch"

    def test_missing_bundle_manifest_is_reported_as_an_index_error(
        self, built: SearchOutcome, session: WorkspaceSession
    ) -> None:
        artifact = session.repository.artifact(built.generation_id)
        assert artifact is not None
        directory = session.paths.indexes.parent / artifact.directory
        (directory / "artifacts.json").unlink()
        with pytest.raises(AtlasReviewError) as caught:
            session.search("prompt injection")
        assert caught.value.code == "missing_artifact_manifest"
        assert caught.value.stage == "index_integrity"

    def test_missing_keyword_artifact_is_rejected(
        self, built: SearchOutcome, session: WorkspaceSession
    ) -> None:
        artifact = session.repository.artifact(built.generation_id)
        assert artifact is not None
        directory = session.paths.indexes.parent / artifact.directory
        (directory / "keyword" / "params.index.json").unlink()
        with pytest.raises(AtlasReviewError) as caught:
            session.search("prompt injection")
        assert caught.value.code == "missing_index_artifact"

    def test_changed_query_settings_are_rejected(
        self, built: SearchOutcome, session: WorkspaceSession
    ) -> None:
        changed = AppConfig.model_validate(
            {
                **session.config.model_dump(mode="python"),
                "keyword_index": {
                    **session.config.keyword_index.model_dump(mode="python"),
                    "k1": 1.9,
                },
            },
            strict=False,
        )
        with pytest.raises(AtlasReviewError) as caught:
            load_generation(
                session.repository,
                passages=session.passages,
                indexes_dir=session.paths.indexes,
                config=changed,
                generation_id=built.generation_id,
            )
        assert caught.value.code == "query_configuration_mismatch"

    @pytest.mark.usefixtures("built")
    def test_foreign_key_in_index_is_rejected(self, session: WorkspaceSession) -> None:
        loaded = _loaded(session)
        broken = LoadedGeneration(
            generation_id=loaded.generation_id,
            collection_version=loaded.collection_version,
            passage_ids=loaded.passage_ids,
            passages={},
            dense_index=loaded.dense_index,
            keyword_index=loaded.keyword_index,
            graph=loaded.graph,
        )
        with pytest.raises(AtlasReviewError) as caught:
            search(broken, RecordingEmbedder(), session.config, "prompt injection")
        assert caught.value.code == "foreign_index_key"
