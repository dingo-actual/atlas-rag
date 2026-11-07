"""Rendering behavior: sanitization and inspection output for untrusted text."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console, RenderableType

from atlas_review.contracts import (
    Candidate,
    ChannelHit,
    GraphHop,
    GraphPath,
    GraphSeed,
    RetrievalTrace,
    StageTiming,
)
from atlas_review.corpus.ingest import IngestionOutcome, IngestionPlan
from atlas_review.services.ingestion import preview_source
from atlas_review.services.session import WorkspaceSession
from atlas_review.tui.render import (
    render_build_outcome,
    render_generations,
    render_ingest_plan,
    render_inspection,
    render_preview,
    render_retrieval_trace,
    sanitize_terminal_text,
)


def _text(renderable: RenderableType) -> str:
    """Render one Rich renderable to plain text without colors."""
    console = Console(record=True, width=120, color_system=None)
    console.print(renderable)
    return console.export_text()


class TestSanitization:
    """Terminal controls never reach the terminal as escape sequences."""

    def test_escape_sequences_are_removed(self) -> None:
        assert sanitize_terminal_text("a\x1b[31mred\x1b[0mb") == "aredb"

    def test_control_characters_are_removed_but_newlines_survive(self) -> None:
        assert sanitize_terminal_text("one\ntwo\x07\x00") == "one\ntwo"


class TestPreviewRendering:
    """A source preview shows identity, counts, and sanitized warnings."""

    def test_preview_includes_digest_counts_and_warnings(
        self, source_bytes: bytes
    ) -> None:
        corpus, preview = preview_source(
            source_bytes, origin="/fixtures/mini_atlas_v6.yaml"
        )
        rendered = _text(render_preview(preview))
        assert "/fixtures/mini_atlas_v6.yaml" in rendered
        assert corpus.snapshot.source_sha256 in rendered
        assert "Apache-2.0" in rendered
        assert "technique=3" in rendered
        assert "employs=3" in rendered
        assert "warnings" in rendered


class TestInspectionRendering:
    """Inspection output carries entity fields, steps, and provenance."""

    def test_case_inspection_shows_steps_and_locators(
        self, stored_session: WorkspaceSession
    ) -> None:
        rendered = _text(render_inspection(stored_session.inspect("AML.CS0035")))
        assert "AML.CS0035" in rendered
        assert "case_study" in rendered
        assert "S01" in rendered
        assert "AML.T0051.001" in rendered

    def test_mitigation_links_show_both_directions(
        self, stored_session: WorkspaceSession
    ) -> None:
        rendered = _text(render_inspection(stored_session.inspect("AML.M0033")))
        assert "Outgoing relationships" in rendered
        assert "mitigates" in rendered
        assert "AML.T0051" in rendered

    def test_generation_table_lists_stored_generation(
        self, stored_session: WorkspaceSession
    ) -> None:
        rendered = _text(render_generations(stored_session.generations()))
        assert stored_session.require_generation_id() in rendered
        assert "building" in rendered

    def test_empty_generation_list_explains_the_next_step(self) -> None:
        rendered = _text(render_generations(()))
        assert "No corpus generation" in rendered


def test_rendering_does_not_modify_stored_workspace(
    stored_session: WorkspaceSession, workspace_root: Path
) -> None:
    before = (workspace_root / "atlas.sqlite3").stat().st_size
    _text(render_inspection(stored_session.inspect("AML.T0051")))
    assert (workspace_root / "atlas.sqlite3").stat().st_size == before


PLAN = IngestionPlan(
    generation_id="generation-1",
    settings_fingerprint="a" * 64,
    chunk_count=12,
    cached_context_count=4,
    uncached_context_count=8,
    reuse_generation=False,
    endpoint_host="provider\x1b[31m.example",
    model="fake-model\x07",
    context_output_tokens=256,
)

OUTCOME = IngestionOutcome(
    generation_id="generation-1",
    passage_count=12,
    context_count=12,
    cached_context_count=4,
    requested_context_count=8,
    dense_count=12,
    keyword_count=12,
    reused_generation=False,
    artifact_directory="indexes/generation-1-bundle",
)


class TestBuildRendering:
    """The plan and the activation summary stay sanitized and explicit."""

    def test_plan_shows_request_count_and_endpoint_without_escapes(self) -> None:
        rendered = _text(render_ingest_plan(PLAN))
        assert "provider.example" in rendered
        assert "fake-model" in rendered
        assert "8" in rendered
        assert "\x1b" not in rendered
        assert "\x07" not in rendered
        assert "credential" in rendered

    def test_reused_plan_states_no_new_requests(self) -> None:
        reused = IngestionPlan(
            generation_id="generation-1",
            settings_fingerprint="a" * 64,
            chunk_count=12,
            cached_context_count=12,
            uncached_context_count=0,
            reuse_generation=True,
            endpoint_host="provider.example",
            model="fake-model",
            context_output_tokens=256,
        )
        rendered = _text(render_ingest_plan(reused))
        assert "already built" in rendered
        assert "credential" not in rendered

    def test_outcome_reports_counts_and_bundle(self) -> None:
        rendered = _text(render_build_outcome(OUTCOME))
        assert "generation-1" in rendered
        assert "indexes/generation-1-bundle" in rendered
        assert "12" in rendered


TRACE = RetrievalTrace(
    schema_version=1,
    generation_id="generation-1",
    collection_version="2026.09",
    settings_fingerprint="a" * 64,
    started_at="2026-11-01T00:00:00+00:00",
    focus="indirect prompt injection\x1b[31m",
    query_tokens=9,
    stage_timings=[
        StageTiming(stage="model_load", seconds=0.5),
        StageTiming(stage="rerank", seconds=0.02),
    ],
    dense_hits=[ChannelHit(passage_id=7, rank=1, value=0.12)],
    keyword_hits=[ChannelHit(passage_id=7, rank=1, value=2.5)],
    graph_seeds=[
        GraphSeed(
            entity_id="AML.T0051.001",
            origin="exact_id",
            fused_rank=None,
            channels=["exact_id"],
        )
    ],
    graph_paths=[
        GraphPath(
            seed_entity_id="AML.T0051.001",
            entity_id="AML.M0033",
            hops=[
                GraphHop(
                    edge_id="e" * 64,
                    source_id="AML.T0051.001",
                    target_id="AML.T0051",
                    relation="specializes",
                    traversed_inverse=False,
                ),
                GraphHop(
                    edge_id="f" * 64,
                    source_id="AML.M0033",
                    target_id="AML.T0051",
                    relation="mitigates",
                    traversed_inverse=True,
                ),
            ],
        )
    ],
    direct_candidate_count=1,
    graph_candidate_count=1,
    candidates=[
        Candidate(
            passage_id=7,
            channels=["dense", "bm25"],
            dense_rank=1,
            dense_distance=0.12,
            bm25_rank=1,
            bm25_score=2.5,
            fusion_score=0.032,
            reranker_score=0.75,
            selected=True,
        )
    ],
    evidence_ids=["E001"],
    exclusions=["p9: evidence budget"],
    limits_reached=["graph_max_edges"],
    missing_categories=["No mitigation description included"],
    provenance=[],
)


class TestTraceRendering:
    """The trace shows seeds, typed paths, ranking, and coverage gaps."""

    def test_trace_renders_paths_without_escapes(self) -> None:
        rendered = _text(render_retrieval_trace(TRACE))
        assert "Retrieval trace" in rendered
        assert "Dense channel" in rendered
        assert "Keyword channel" in rendered
        assert "Reranked candidates" in rendered
        assert "dense+bm25" in rendered
        assert "AML.T0051.001" in rendered
        assert "specializes" in rendered
        assert "mitigates" in rendered
        assert "0.7500" in rendered
        assert "yes" in rendered
        assert "graph_max_edges" in rendered
        assert "excluded: p9: evidence budget" in rendered
        assert "model_load=0.500" in rendered
        assert "keyword zero-score exclusions: 0" in rendered
        assert "\x1b" not in rendered

    def test_empty_trace_states_no_expansion(self) -> None:
        empty = RetrievalTrace(
            schema_version=1,
            generation_id="generation-1",
            collection_version="2026.09",
            settings_fingerprint="a" * 64,
            started_at="2026-11-01T00:00:00+00:00",
            focus="nothing matches",
            query_tokens=2,
        )
        rendered = _text(render_retrieval_trace(empty))
        assert "no graph expansion" in rendered
        assert "no candidates" in rendered
        assert "evidence: none" in rendered
