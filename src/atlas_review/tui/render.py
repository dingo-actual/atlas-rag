"""Terminal rendering for source previews and corpus inspection.

Rendering stays presentation-only: it reads service results and never queries
SQLite, scores records, or builds prompts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_review.config import (
    PLACEHOLDER_BASE_URL,
    PLACEHOLDER_MODEL,
    LLMConfig,
)
from atlas_review.contracts import (
    AtlasEntity,
    AtlasRelationship,
    Candidate,
    GraphPath,
    RetrievalTrace,
    StageTiming,
)
from atlas_review.corpus.ingest import IngestionOutcome, IngestionPlan
from atlas_review.errors import AtlasReviewError
from atlas_review.sanitize import sanitize_text
from atlas_review.services.ingestion import SourcePreview
from atlas_review.storage.repository import EntityInspection, GenerationSummary


def sanitize_terminal_text(value: str) -> str:
    """Strip control characters and escape sequences from untrusted text.

    Args:
        value (str): Text that may contain ATLAS or model content.

    Returns:
        (str) Text safe to render in a terminal.
    """
    return sanitize_text(value)


def endpoint_host(base_url: str | None) -> str:
    """Return the host a request would reach, for display only.

    Args:
        base_url (str | None): Configured API root.

    Returns:
        (str) Hostname, or a notice when no endpoint is configured.
    """
    if base_url is None:
        return "not configured"
    return urlsplit(base_url).hostname or "not configured"


def inference_label(config: LLMConfig) -> str:
    """Render the configured endpoint and model for the status header.

    The documented placeholder provider is reported as not configured.

    Args:
        config (LLMConfig): Stored provider settings.

    Returns:
        (str) Endpoint host and model, or a not-configured notice.
    """
    if (config.base_url, config.model) == (PLACEHOLDER_BASE_URL, PLACEHOLDER_MODEL):
        return "not configured"
    return f"{endpoint_host(config.base_url)} / {sanitize_text(config.model)}"


def render_status(
    *, workspace: Path, collection: str, inference: str
) -> RenderableType:
    """Render the main-screen header of workspace, corpus, and endpoint state.

    Args:
        workspace (Path): Workspace directory in use.
        collection (str): Active ATLAS collection version or a notice.
        inference (str): Endpoint and model, or a notice.

    Returns:
        (RenderableType) Compact status panel.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("workspace", sanitize_text(str(workspace)))
    table.add_row("corpus", sanitize_text(collection))
    table.add_row("inference", sanitize_text(inference))
    return Panel(table, title="atlas-review", border_style="cyan")


def render_menu(choices: Sequence[tuple[str, str]]) -> RenderableType:
    """Render the numbered main menu.

    Args:
        choices (Sequence[tuple[str, str]]): Key and label pairs.

    Returns:
        (RenderableType) Menu panel.
    """
    lines = [f"{key}  {label}" for key, label in choices]
    return Panel(
        Text("\n".join(lines), overflow="fold"),
        title="Main menu",
        border_style="blue",
    )


def print_error(console: Console, error: AtlasReviewError) -> None:
    """Render an application failure without exposing stored payloads.

    Args:
        console (Console): Rich console.
        error (AtlasReviewError): Expected application failure.
    """
    body = [f"{error.message}", f"code: {error.code}  stage: {error.stage}"]
    body.extend(error.details)
    if error.recovery:
        body.append(f"next: {error.recovery}")
    console.print(Panel("\n".join(body), title="Error", border_style="red"))


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
    table.add_row("acquired", sanitize_terminal_text(preview.acquired_at))
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


def render_retrieval_trace(trace: RetrievalTrace) -> RenderableType:
    """Render one local retrieval run: seeds, paths, ranking, and evidence.

    Args:
        trace (RetrievalTrace): Validated diagnostic record of the run.

    Returns:
        (RenderableType) Grouped tables for channels, graph, and evidence.
    """
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", no_wrap=True)
    header.add_column(overflow="fold")
    header.add_row("generation", trace.generation_id)
    header.add_row("collection", sanitize_terminal_text(trace.collection_version))
    header.add_row("settings", trace.settings_fingerprint[:12])
    header.add_row("started", sanitize_terminal_text(trace.started_at))
    header.add_row("focus", sanitize_terminal_text(trace.focus))
    header.add_row("query tokens", str(trace.query_tokens))
    header.add_row("stage seconds", _timings(trace.stage_timings))
    header.add_row(
        "direct candidates",
        f"{trace.direct_candidate_count} of {trace.direct_quota_before_dedup} offered",
    )
    header.add_row(
        "graph candidates",
        f"{trace.graph_candidate_count} of {trace.graph_quota_before_dedup} offered",
    )
    return Group(
        Panel(header, title="Retrieval trace", border_style="cyan"),
        _channel_table(
            "Dense channel",
            [(hit.passage_id, hit.value) for hit in trace.dense_hits],
            "distance",
        ),
        _channel_table(
            "Keyword channel",
            [(hit.passage_id, hit.value) for hit in trace.keyword_hits],
            "score",
        ),
        _path_table(trace.graph_paths),
        _candidate_table(trace.candidates),
        _notes_panel(trace),
    )


def _timings(timings: Sequence[StageTiming]) -> str:
    """Render stage durations as a compact one-line list."""
    if not timings:
        return "none"
    return ", ".join(f"{timing.stage}={timing.seconds:.3f}" for timing in timings)


def _path_table(paths: Sequence[GraphPath]) -> RenderableType:
    """Render each discovered seed-to-entity path as a readable walk."""
    if not paths:
        return Panel(
            Text("no graph expansion", style="dim"),
            title="Graph paths",
            border_style="blue",
        )
    table = Table(title="Graph paths", title_justify="left")
    table.add_column("seed")
    table.add_column("hops")
    table.add_column("reached")
    for path in paths:
        walk = " -> ".join(
            [path.seed_entity_id]
            + [
                f"{'<-' if hop.traversed_inverse else '-'}{hop.relation}->"
                f" {hop.source_id if hop.traversed_inverse else hop.target_id}"
                for hop in path.hops
            ]
        )
        table.add_row(
            sanitize_terminal_text(path.seed_entity_id),
            Text(walk, overflow="fold"),
            sanitize_terminal_text(path.entity_id),
        )
    return Panel(table, border_style="blue")


def _candidate_table(candidates: Sequence[Candidate]) -> RenderableType:
    """Render the reranked pool with channel and selection provenance."""
    if not candidates:
        return Panel(
            Text("no candidates", style="dim"),
            title="Reranked candidates",
            border_style="yellow",
        )
    table = Table(title="Reranked candidates", title_justify="left")
    for header in ("rank", "passage", "channels", "score", "selected", "graph"):
        table.add_column(header, overflow="fold")
    for rank, candidate in enumerate(candidates, start=1):
        score = (
            f"{candidate.reranker_score:.4f}"
            if candidate.reranker_score is not None
            else "-"
        )
        graph = "; ".join(
            " -> ".join([path.seed_entity_id, *(hop.edge_id[:8] for hop in path.hops)])
            for path in candidate.graph_paths
        )
        table.add_row(
            str(rank),
            str(candidate.passage_id),
            "+".join(candidate.channels),
            score,
            "yes" if candidate.selected else "no",
            Text(graph or "-", overflow="fold"),
        )
    return Panel(table, border_style="yellow")


def _notes_panel(trace: RetrievalTrace) -> RenderableType:
    """Render evidence identifiers, exclusions, caps, and coverage gaps."""
    body = [
        f"evidence: {', '.join(trace.evidence_ids) or 'none'}",
        f"keyword zero-score exclusions: {trace.keyword_zero_score_exclusions}",
        f"limits reached: {', '.join(trace.limits_reached) or 'none'}",
        f"coverage gaps: {'; '.join(trace.missing_categories) or 'none'}",
    ]
    body.extend(f"excluded: {line}" for line in trace.exclusions)
    return Panel(
        Text("\n".join(body), overflow="fold"),
        title="Evidence",
        border_style="green",
    )


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
