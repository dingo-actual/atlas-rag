"""Terminal rendering for source previews and corpus inspection.

Rendering stays presentation-only: it reads service results and never queries
SQLite, scores records, or builds prompts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_review.contracts import AtlasEntity, AtlasRelationship
from atlas_review.corpus.ingest import IngestionOutcome, IngestionPlan
from atlas_review.retrieval.hybrid import SearchOutcome
from atlas_review.services.ingestion import SourcePreview
from atlas_review.storage.repository import EntityInspection, GenerationSummary

_TERMINAL_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_ESCAPE_SEQUENCES = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def sanitize_terminal_text(value: str) -> str:
    """Strip control characters and escape sequences from untrusted text.

    Args:
        value (str): Text that may contain ATLAS or model content.

    Returns:
        (str) Text safe to render in a terminal.
    """
    without_escapes = _ESCAPE_SEQUENCES.sub("", value)
    return _TERMINAL_CONTROLS.sub("", without_escapes)


def render_preview(preview: SourcePreview) -> Panel:
    """Render a stored source preview as one panel.

    Args:
        preview (SourcePreview): Parsed-source summary from the service layer.

    Returns:
        (Panel) Panel describing the source identity and record counts.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("origin", sanitize_terminal_text(preview.origin))
    table.add_row("bytes", str(preview.byte_count))
    table.add_row("sha256", preview.source_sha256)
    table.add_row("format", sanitize_terminal_text(preview.format_version))
    table.add_row("collection", sanitize_terminal_text(preview.collection_version))
    table.add_row(
        "license", sanitize_terminal_text(preview.license_id or "unspecified")
    )
    table.add_row("entities", _counts(preview.entity_counts))
    table.add_row("edges", _counts(preview.edge_counts))
    body: list[RenderableType] = [table]
    if preview.warnings:
        body.append(Text("warnings", style="bold yellow"))
        body.extend(
            Text(f"- {sanitize_terminal_text(warning)}", style="yellow")
            for warning in preview.warnings
        )
    return Panel(Group(*body), title="Source preview", border_style="cyan")


def render_generations(summaries: Sequence[GenerationSummary]) -> RenderableType:
    """Render stored generations newest first as a table.

    Args:
        summaries (Sequence[GenerationSummary]): Repository summaries.

    Returns:
        (RenderableType) Table, or a notice when nothing is stored.
    """
    if not summaries:
        return Text("No corpus generation is stored yet.", style="yellow")
    table = Table(title="Stored generations", title_justify="left")
    headers = ("generation", "created", "collection", "status", "entities", "edges")
    for header in headers:
        table.add_column(header, overflow="fold")
    for summary in summaries:
        table.add_row(
            summary.generation_id,
            sanitize_terminal_text(summary.created_at),
            sanitize_terminal_text(summary.collection_version),
            summary.status,
            str(summary.entity_count),
            str(summary.edge_count),
        )
    return table


def render_inspection(inspection: EntityInspection) -> RenderableType:
    """Render one entity with both relationship directions and provenance.

    Args:
        inspection (EntityInspection): Entity, outgoing edges, and incoming
            edges from the repository.

    Returns:
        (RenderableType) Grouped renderables.
    """
    entity = inspection.entity
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", no_wrap=True)
    header.add_column(overflow="fold")
    header.add_row("entity", entity.entity_id)
    header.add_row("kind", entity.kind)
    header.add_row("name", sanitize_terminal_text(entity.display_name))
    header.add_row("locator", sanitize_terminal_text(entity.source_locator))
    header.add_row("description", _wrapped(entity.description))
    if entity.references:
        header.add_row("references", _references(entity))
    return Group(
        Panel(header, title="Entity", border_style="green"),
        _edges_table("Outgoing relationships", inspection.outgoing),
        _edges_table("Incoming relationships", inspection.incoming),
    )


def render_ingest_plan(plan: IngestionPlan) -> Panel:
    """Render the LLM-free build plan before any provider request.

    Args:
        plan (IngestionPlan): Planned chunk, cache, and endpoint facts.

    Returns:
        (Panel) Panel stating what the build will do and what it will cost.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("generation", plan.generation_id)
    table.add_row("source chunks", str(plan.chunk_count))
    table.add_row("cached context", str(plan.cached_context_count))
    table.add_row("provider requests", str(plan.uncached_context_count))
    table.add_row(
        "endpoint",
        sanitize_terminal_text(plan.endpoint_host or "not configured"),
    )
    table.add_row("model", sanitize_terminal_text(plan.model))
    table.add_row("output tokens per request", str(plan.context_output_tokens))
    body: list[RenderableType] = [table]
    if plan.reuse_generation:
        body.append(
            Text(
                "An identical generation is already built; it will be activated "
                "without new requests.",
                style="green",
            )
        )
    elif plan.uncached_context_count:
        body.append(
            Text(
                "Building sends source excerpts to the configured endpoint and "
                "needs its credential environment variable.",
                style="yellow",
            )
        )
    return Panel(Group(*body), title="Build plan", border_style="magenta")


def render_build_outcome(outcome: IngestionOutcome) -> Panel:
    """Render the activation summary of one explicit build.

    Args:
        outcome (IngestionOutcome): Activation summary from the service layer.

    Returns:
        (Panel) Panel with counts and the published bundle path.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("generation", outcome.generation_id)
    table.add_row("passages", str(outcome.passage_count))
    table.add_row("context entries", str(outcome.context_count))
    table.add_row("reused context", str(outcome.cached_context_count))
    table.add_row("provider requests", str(outcome.requested_context_count))
    table.add_row("dense vectors", str(outcome.dense_count))
    table.add_row("keyword documents", str(outcome.keyword_count))
    table.add_row("reused generation", "yes" if outcome.reused_generation else "no")
    table.add_row("bundle", sanitize_terminal_text(outcome.artifact_directory))
    return Panel(table, title="Ingestion complete", border_style="green")


def render_search(outcome: SearchOutcome) -> RenderableType:
    """Render both search channels and the fused candidate list.

    Args:
        outcome (SearchOutcome): Channel hits, fused candidates, and passages.

    Returns:
        (RenderableType) Grouped tables for one retrieval run.
    """
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", no_wrap=True)
    header.add_column(overflow="fold")
    header.add_row("generation", outcome.generation_id)
    header.add_row("collection", sanitize_terminal_text(outcome.collection_version))
    header.add_row("focus", sanitize_terminal_text(outcome.focus))
    return Group(
        Panel(header, title="Retrieval", border_style="cyan"),
        _channel_table("Dense channel", outcome.dense_hits, "distance"),
        _channel_table("Keyword channel", outcome.keyword_hits, "score"),
        _fused_table(outcome),
    )


def _channel_table(
    title: str, hits: Sequence[tuple[int, float]], label: str
) -> RenderableType:
    """Render one channel's ranked ``(passage_id, value)`` hits."""
    if not hits:
        return Panel(
            Text("no candidates", style="dim"), title=title, border_style="blue"
        )
    table = Table(title=title, title_justify="left")
    table.add_column("rank")
    table.add_column("passage")
    table.add_column(label)
    for rank, (passage_id, value) in enumerate(hits, start=1):
        table.add_row(str(rank), str(passage_id), f"{value:.4f}")
    return Panel(table, border_style="blue")


def _fused_table(outcome: SearchOutcome) -> RenderableType:
    """Render fused candidates with their passage headers."""
    if not outcome.candidates:
        return Panel(
            Text("no candidates", style="dim"),
            title="Fused candidates",
            border_style="yellow",
        )
    table = Table(title="Fused candidates", title_justify="left", show_lines=True)
    headers = ("rank", "passage", "channels", "fusion", "header")
    for header in headers:
        table.add_column(header, overflow="fold")
    for rank, (candidate, passage) in enumerate(
        zip(outcome.candidates, outcome.passages, strict=True), start=1
    ):
        table.add_row(
            str(rank),
            str(candidate.passage_id),
            "+".join(candidate.channels),
            f"{candidate.fusion_score:.5f}",
            _wrapped(passage.header),
        )
    return Panel(table, border_style="yellow")


def _counts(counts: Sequence[tuple[str, int]]) -> str:
    """Render count pairs as ``kind=count`` text."""
    return ", ".join(f"{kind}={count}" for kind, count in counts) or "none"


def _wrapped(text: str) -> Text:
    """Wrap untrusted multi-line text for display inside a table cell."""
    return Text(sanitize_terminal_text(text), overflow="fold")


def _references(entity: AtlasEntity) -> Text:
    """Render bibliography entries without following or opening links."""
    lines = [
        f"[{reference.ref_id}] {reference.title}"
        + (f" <{reference.url}>" if reference.url is not None else "")
        for reference in entity.references
    ]
    return _wrapped("\n".join(lines))


def _edges_table(title: str, edges: Sequence[AtlasRelationship]) -> RenderableType:
    """Render edge records with their step, tactic, and source provenance."""
    if not edges:
        return Panel(Text("none", style="dim"), title=title, border_style="blue")
    table = Table(title=title, title_justify="left", show_lines=True)
    headers = ("relation", "direction", "step", "tactic", "locator", "description")
    for header in headers:
        table.add_column(header, overflow="fold")
    for edge in edges:
        table.add_row(
            edge.relation,
            f"{edge.source_id} -> {edge.target_id}",
            edge.step_id or "-",
            edge.tactic_id or "-",
            sanitize_terminal_text(edge.source_locator),
            _wrapped(edge.description or ""),
        )
    return Panel(table, border_style="blue")
