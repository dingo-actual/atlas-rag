"""Terminal rendering for evidence inspection, reviews, and inference settings."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas_review.citations import case_label, mitigation_labels
from atlas_review.config import LLMConfig
from atlas_review.contracts import (
    ContextualizationResult,
    ControlRecommendation,
    EvidenceItem,
    EvidencePacket,
    GraphPath,
    Passage,
    ReviewAttempt,
    ReviewFinding,
)
from atlas_review.storage.reviews import ReviewSummary, StoredReview
from atlas_review.tui.render import endpoint_host, sanitize_terminal_text


def render_evidence_packet(
    packet: EvidencePacket,
    *,
    missing_categories: Sequence[str] = (),
    exclusions: Sequence[str] = (),
) -> RenderableType:
    """Render the frozen evidence packet with its coverage notices.

    Args:
        packet (EvidencePacket): Evidence packet to display.
        missing_categories (Sequence[str]): Categories without stored ATLAS text.
        exclusions (Sequence[str]): Evidence units that did not fit the budget.

    Returns:
        (RenderableType) Packet panel with one row per evidence item.
    """
    table = Table(box=None, pad_edge=False)
    for column in ("ID", "Kind", "Entities", "Case", "Channels"):
        table.add_column(column)
    for item in packet.items:
        table.add_row(
            sanitize_terminal_text(item.evidence_id),
            item.item_kind,
            sanitize_terminal_text(", ".join(item.entity_ids)),
            sanitize_terminal_text(item.case_type or "-"),
            sanitize_terminal_text(", ".join(item.retrieval_channels)),
        )
    notices = [
        Text(f"Warning: no ATLAS text for {name}.", style="yellow")
        for name in missing_categories
    ]
    notices.extend(
        Text(f"Excluded from the packet: {name}.", style="yellow")
        for name in exclusions
    )
    body = Group(
        Text(
            f"{len(packet.items)} items, collection "
            f"{sanitize_terminal_text(packet.collection_version)}",
            style="bold",
        ),
        *notices,
        table,
        Text(
            "Generated retrieval context is excluded from every excerpt.",
            style="dim",
        ),
    )
    return Panel(body, title="Evidence packet", border_style="green")


def render_review_header(title: str, focus: str) -> RenderableType:
    """Render the review title and the focus the evidence was retrieved for.

    Args:
        title (str): Brief title supplied by the reviewer.
        focus (str): Retrieval focus used for this run.

    Returns:
        (RenderableType) Review panel header.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("title", sanitize_terminal_text(title))
    table.add_row("focus", sanitize_terminal_text(focus))
    return Panel(table, title="Review", border_style="blue")


def render_evidence_detail(item: EvidenceItem) -> RenderableType:
    """Render one evidence item with its source text and provenance.

    Args:
        item (EvidenceItem): Stored evidence item.

    Returns:
        (RenderableType) Item panel.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("id", sanitize_terminal_text(item.evidence_id))
    table.add_row("kind", item.item_kind)
    table.add_row("entities", sanitize_terminal_text(", ".join(item.entity_ids)))
    table.add_row("case type", sanitize_terminal_text(item.case_type or "-"))
    table.add_row("locator", sanitize_terminal_text(item.source_locator))
    table.add_row(
        "channels", sanitize_terminal_text(", ".join(item.retrieval_channels))
    )
    if item.passage_id is not None:
        table.add_row("passage", str(item.passage_id))
    if item.edge_id is not None:
        table.add_row("edge", sanitize_terminal_text(item.edge_id))
    body: list[RenderableType] = [
        table,
        Text(""),
        Text(sanitize_terminal_text(item.text), overflow="fold"),
    ]
    if item.graph_path is not None:
        body.extend((Text(""), _path_table(item.graph_path)))
    return Panel(Group(*body), title="Evidence item", border_style="green")


def _path_table(path: GraphPath) -> RenderableType:
    """Render one evidence graph path as original edges and directions."""
    table = Table(title="Graph path", title_justify="left")
    for column in ("source", "relationship", "original target", "direction"):
        table.add_column(column, overflow="fold")
    for hop in path.hops:
        table.add_row(
            sanitize_terminal_text(hop.source_id),
            hop.relation,
            sanitize_terminal_text(hop.target_id),
            "inverse" if hop.traversed_inverse else "forward",
        )
    return table


def render_generated_context(
    passage: Passage, entry: ContextualizationResult | None
) -> RenderableType:
    """Render the generated retrieval note beside its original excerpt.

    Args:
        passage (Passage): Stored passage holding both text views.
        entry (ContextualizationResult | None): Accepted cache entry behind the
            note, when it is still stored.

    Returns:
        (RenderableType) Labelled comparison of the note and the source text.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("passage", str(passage.passage_id))
    table.add_row("cache key", sanitize_terminal_text(passage.context_cache_key[:12]))
    if entry is not None:
        table.add_row("prompt version", sanitize_terminal_text(entry.prompt_version))
        table.add_row("model", sanitize_terminal_text(entry.model))
        table.add_row(
            "support locators",
            sanitize_terminal_text(", ".join(entry.support_locators)),
        )
    body = Group(
        Text(
            "LLM-generated retrieval aid. It is not ATLAS source text and never "
            "enters a citation or the synthesis packet.",
            style="yellow",
            overflow="fold",
        ),
        Text(""),
        table,
        Text(""),
        Text("Generated note", style="bold"),
        Text(sanitize_terminal_text(passage.context_text), overflow="fold"),
        Text(""),
        Text("Original source excerpt", style="bold"),
        Text(sanitize_terminal_text(passage.source_text), overflow="fold"),
    )
    return Panel(body, title="Generated context", border_style="magenta")


def render_request_summary(config: LLMConfig, item_count: int) -> RenderableType:
    """Render the request about to be sent, before any provider call.

    Args:
        config (LLMConfig): Provider settings for the request.
        item_count (int): Evidence items the request will carry.

    Returns:
        (RenderableType) Request summary panel.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("destination", endpoint_host(config.base_url))
    table.add_row("model", sanitize_terminal_text(config.model))
    table.add_row("evidence items", str(item_count))
    table.add_row("output tokens", str(config.output_tokens))
    table.add_row("max findings", str(config.max_findings))
    body = Group(
        table,
        Text(""),
        Text(
            "This request sends the brief and the listed ATLAS evidence to "
            "the configured endpoint.",
            style="yellow",
            overflow="fold",
        ),
    )
    return Panel(body, title="Review request", border_style="magenta")


def render_review(
    stored: StoredReview, attempts: Sequence[ReviewAttempt] = ()
) -> RenderableType:
    """Render a saved review with its report, attempts, and provenance.

    Args:
        stored (StoredReview): Saved review.
        attempts (Sequence[ReviewAttempt]): Recorded attempts for this review.

    Returns:
        (RenderableType) Report panel or a not-yet-generated notice.
    """
    header = _review_header(stored)
    if stored.report is None:
        body = Group(
            header,
            Text(""),
            Text(
                "No validated report is stored. The evidence packet is frozen; "
                "choose Generate to make one explicit provider request.",
                overflow="fold",
            ),
            *_attempts_block(attempts),
        )
        return Panel(body, title="Saved review", border_style="yellow")
    report = stored.report
    pieces: list[RenderableType] = [
        header,
        Text(""),
        Text("Summary", style="bold"),
        Text(sanitize_terminal_text(report.summary), overflow="fold"),
    ]
    pieces.extend(
        _finding_panel(finding, stored, position)
        for position, finding in enumerate(report.findings, start=1)
    )
    if not report.findings:
        pieces.append(
            Text(
                "No findings: the stored evidence did not support a review concern.",
                overflow="fold",
            )
        )
    pieces.extend(_listed_texts("Unanswered questions", report.unanswered_questions))
    pieces.extend(_listed_texts("Evidence gaps", report.evidence_gaps))
    pieces.extend(_attempts_block(attempts))
    return Panel(Group(*pieces), title="Saved review", border_style="green")


def render_reviews(summaries: Sequence[ReviewSummary]) -> RenderableType:
    """Render the saved-review list.

    Args:
        summaries (Sequence[ReviewSummary]): Saved reviews newest first.

    Returns:
        (RenderableType) Review table or a notice when the list is empty.
    """
    if not summaries:
        return Panel(
            "No saved reviews yet.", title="Saved reviews", border_style="yellow"
        )
    table = Table(box=None, pad_edge=False)
    for column in ("Review", "Created", "Title", "State", "Collection"):
        table.add_column(column)
    for summary in summaries:
        table.add_row(
            sanitize_terminal_text(summary.review_id),
            sanitize_terminal_text(summary.created_at),
            sanitize_terminal_text(summary.title),
            summary.state,
            sanitize_terminal_text(summary.collection_version),
        )
    return Panel(table, title="Saved reviews", border_style="blue")


def render_configuration(
    config: LLMConfig, *, credential_present: bool
) -> RenderableType:
    """Render the configured inference settings without any secret.

    Args:
        config (LLMConfig): Provider settings to display.
        credential_present (bool): Whether the named variable is currently set.

    Returns:
        (RenderableType) Configuration panel.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("provider", sanitize_terminal_text(config.provider_name))
    table.add_row("endpoint", endpoint_host(config.base_url))
    table.add_row("model", sanitize_terminal_text(config.model))
    table.add_row("API key mode", config.api_key_mode)
    table.add_row(
        "API key variable",
        sanitize_terminal_text(config.api_key_env or "none"),
    )
    table.add_row(
        "variable set",
        "not needed"
        if config.api_key_mode == "none"
        else ("yes" if credential_present else "no"),
    )
    table.add_row("output tokens", str(config.output_tokens))
    table.add_row("output parameter", config.output_token_parameter)
    table.add_row("response format", config.response_format_mode)
    table.add_row("max findings", str(config.max_findings))
    table.add_row(
        "temperature",
        "provider default" if config.temperature is None else str(config.temperature),
    )
    table.add_row("timeout", f"{config.timeout_seconds:g}s")
    table.add_row("request budget", f"{config.max_input_characters} characters")
    return Panel(table, title="Inference configuration", border_style="blue")


def _review_header(stored: StoredReview) -> RenderableType:
    """Render trusted provenance and the untrusted title."""
    provider = stored.config.llm
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("title", sanitize_terminal_text(stored.title))
    table.add_row("review", sanitize_terminal_text(stored.review_id))
    table.add_row("state", stored.state)
    table.add_row("created", sanitize_terminal_text(stored.created_at))
    table.add_row(
        "collection", sanitize_terminal_text(stored.evidence.collection_version)
    )
    table.add_row("retrieval model", sanitize_terminal_text(provider.model))
    table.add_row("source", "supplied brief; not a scan or test result")
    return table


def _finding_panel(
    finding: ReviewFinding, stored: StoredReview, position: int
) -> RenderableType:
    """Render one finding with priority, grounding, controls, and citations."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("priority", finding.priority)
    table.add_row("why", sanitize_terminal_text(finding.priority_rationale))
    table.add_row(
        "techniques", sanitize_terminal_text(", ".join(finding.technique_ids))
    )
    table.add_row("applicability", sanitize_terminal_text(finding.applicability))
    table.add_row("ATLAS basis", sanitize_terminal_text(finding.atlas_basis))
    if finding.case_study_ids:
        table.add_row(
            "case studies", sanitize_terminal_text(_case_labels(finding, stored))
        )
    table.add_row("evidence", sanitize_terminal_text(", ".join(finding.evidence_ids)))
    body: list[RenderableType] = [table]
    for title, values in (
        ("source facts", finding.input_facts),
        ("assumptions and unknowns", finding.assumptions_or_unknowns),
        ("review questions", finding.review_questions),
    ):
        body.extend(_listed_texts(title, values))
    body.extend(
        _control_lines(control, finding, stored) for control in finding.controls
    )
    title = f"{position}. {sanitize_terminal_text(finding.finding_id)}: {sanitize_terminal_text(finding.title)}"
    return Panel(Group(*body), title=title, border_style="white")


def _control_lines(
    control: ControlRecommendation, finding: ReviewFinding, stored: StoredReview
) -> RenderableType:
    """Render one control recommendation with its recorded association kind."""
    mitigations = mitigation_labels(
        control.mitigation_ids,
        finding.technique_ids,
        stored.evidence.allowed_associations,
    )
    text = (
        f"Control: {control.action}\n"
        f"  Rationale: {control.rationale}\n"
        f"  Verification: {control.verification}\n"
        f"  ATLAS mitigations: {mitigations} "
        f"(an association is not evidence of effectiveness)\n"
        f"  Evidence: {', '.join(control.evidence_ids)}"
    )
    return Text(sanitize_terminal_text(text), overflow="fold")


def _case_labels(finding: ReviewFinding, stored: StoredReview) -> str:
    """Return cited cases with their stored classification."""
    return "; ".join(
        case_label(case_id, stored.evidence.items) for case_id in finding.case_study_ids
    )


def _attempts_block(attempts: Sequence[ReviewAttempt]) -> list[RenderableType]:
    """Render the attempt history, or nothing when none was recorded."""
    if not attempts:
        return []
    table = Table(box=None, pad_edge=False)
    for column in ("#", "State", "Started", "Model", "Tokens", "Seconds", "Error"):
        table.add_column(column)
    for attempt in attempts:
        table.add_row(
            str(attempt.attempt_number),
            attempt.state,
            sanitize_terminal_text(attempt.started_at),
            sanitize_terminal_text(attempt.model),
            _token_cell(attempt),
            "-"
            if attempt.duration_seconds is None
            else f"{attempt.duration_seconds:.1f}",
            sanitize_terminal_text(attempt.error_code or "-"),
        )
    return [Text(""), Text("Attempts", style="bold"), table]


def _token_cell(attempt: ReviewAttempt) -> str:
    """Render reported prompt/completion tokens, keeping absent values null."""
    if attempt.usage_input_tokens is None:
        return "-"
    output = (
        "-" if attempt.usage_output_tokens is None else str(attempt.usage_output_tokens)
    )
    return f"{attempt.usage_input_tokens}/{output}"


def _listed_texts(title: str, values: Sequence[str]) -> list[RenderableType]:
    """Render an optional labelled list."""
    if not values:
        return []
    return [
        Text(""),
        Text(title, style="bold"),
        *[
            Text(f"- {sanitize_terminal_text(value)}", overflow="fold")
            for value in values
        ],
    ]
