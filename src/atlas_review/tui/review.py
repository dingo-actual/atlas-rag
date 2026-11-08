"""Review screens: brief entry, evidence inspection, generation, and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from atlas_review.contracts import ReviewRequest
from atlas_review.errors import STAGE_EXPORT, AtlasReviewError, validation_details
from atlas_review.services.session import ReviewDraft, WorkspaceSession
from atlas_review.tui.configuration import edit_configuration
from atlas_review.tui.render import (
    print_error,
    render_retrieval_trace,
    sanitize_terminal_text,
)
from atlas_review.tui.review_render import (
    render_evidence_detail,
    render_evidence_packet,
    render_generated_context,
    render_request_summary,
    render_review,
    render_review_header,
)

BRIEF_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", "Short review title"),
    ("review_question", "Review question"),
    ("retrieval_focus", "Retrieval focus"),
    ("system_description", "System description"),
    ("users_and_roles", "Users and roles"),
    ("data_sources", "Data sources"),
    ("untrusted_inputs", "Untrusted inputs"),
    ("tools_and_permissions", "Tools and permissions"),
    ("sensitive_assets", "Sensitive assets"),
    ("existing_controls", "Existing controls"),
    ("constraints", "Constraints"),
    ("unknowns", "Unknowns"),
)
_LIST_FIELDS = frozenset(
    {
        "users_and_roles",
        "data_sources",
        "untrusted_inputs",
        "tools_and_permissions",
        "sensitive_assets",
        "existing_controls",
        "constraints",
        "unknowns",
    }
)
_TEXT_FIELDS = frozenset(
    {"title", "review_question", "retrieval_focus", "system_description"}
)
_REQUIRED = frozenset(
    {"title", "review_question", "retrieval_focus", "system_description"}
)


def run_review(console: Console, session: WorkspaceSession) -> None:
    """Run the review screen from brief entry to report export.

    Args:
        console (Console): Rich console.
        session (WorkspaceSession): Open workspace session.
    """
    if session.repository.active_generation_id() is None:
        console.print(
            Panel(
                "No active corpus generation. Ingest an ATLAS source first.",
                border_style="yellow",
            )
        )
        return
    draft = _retrieve(console, session)
    if draft is None:
        return
    while True:
        selection = draft.run.selection
        console.print(
            render_review_header(draft.brief.title, draft.brief.retrieval_focus)
        )
        console.print(
            render_evidence_packet(
                selection.packet,
                missing_categories=selection.missing_categories,
                exclusions=selection.exclusions,
            )
        )
        if not _evidence_menu(console, session, draft):
            return
        draft = _retrieve(console, session, draft.brief.model_dump(mode="json"))
        if draft is None:
            return


def _retrieve(
    console: Console,
    session: WorkspaceSession,
    pending: dict[str, Any] | None = None,
) -> ReviewDraft | None:
    """Collect a brief and retrieve evidence, keeping the draft on failure."""
    while True:
        brief = _obtain_brief(console, session, pending)
        if brief is None:
            return None
        try:
            return session.retrieve_review(brief)
        except AtlasReviewError as error:
            print_error(console, error)
            pending = brief.model_dump(mode="json")


def _obtain_brief(
    console: Console, session: WorkspaceSession, draft: dict[str, Any] | None = None
) -> ReviewRequest | None:
    """Collect or load a brief, then validate it strictly."""
    if draft is None:
        source = Prompt.ask(
            "Brief",
            choices=["new", "load", "return"],
            default="new",
            console=console,
        )
        if source == "return":
            return None
        draft = blank_brief()
        if source == "load":
            loaded = _load_brief(console, session.paths.root)
            if loaded is None:
                return None
            draft = loaded.model_dump(mode="json")
    while True:
        console.print(_brief_panel(draft))
        command = Prompt.ask(
            "Field to edit, 'save' to write the draft, 'ready' to retrieve "
            "evidence, or 'return'",
            default="ready",
            console=console,
        ).strip()
        if command == "return":
            return None
        if command == "ready":
            validated = _validate_brief(console, draft)
            if validated is not None:
                return validated
            continue
        if command == "save":
            _save_brief(console, session, draft)
            continue
        if command not in dict(BRIEF_FIELDS):
            console.print(f"Unknown field {command!r}.")
            continue
        updated = _edit_field(console, command, draft.get(command))
        if updated is not None:
            draft[command] = updated


def blank_brief() -> dict[str, Any]:
    """Return an empty brief draft holding only contract fields."""
    draft: dict[str, Any] = {"schema_version": 1}
    for field in _LIST_FIELDS:
        draft[field] = []
    for field in _TEXT_FIELDS:
        draft[field] = ""
    return draft


def save_brief(brief: ReviewRequest, target: Path) -> Path:
    """Write one validated brief as a loadable JSON document.

    Args:
        brief (ReviewRequest): Validated brief to write.
        target (Path): Destination file.

    Returns:
        (Path) Written file.

    Raises:
        AtlasReviewError: The destination cannot be written.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(brief.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise AtlasReviewError(
            code="export_failed",
            stage=STAGE_EXPORT,
            message=f"cannot write this brief in {target.parent}",
            recovery="Choose a writable destination and save again.",
        ) from exc
    return target


def _save_brief(
    console: Console, session: WorkspaceSession, draft: dict[str, Any]
) -> None:
    """Validate the draft and offer to write it as a brief JSON document."""
    brief = _validate_brief(console, draft)
    if brief is None:
        return
    raw = Prompt.ask(
        "Brief JSON path",
        default=str(session.paths.exports / "brief.json"),
        console=console,
    )
    try:
        target = save_brief(brief, Path(raw).expanduser())
    except AtlasReviewError as error:
        print_error(console, error)
        return
    console.print(f"Saved {sanitize_terminal_text(str(target))}")


def _load_brief(console: Console, root: Path) -> ReviewRequest | None:
    """Load and strictly validate a saved brief file."""
    raw = Prompt.ask(
        "Brief JSON path", default=str(root / "brief.json"), console=console
    )
    path = Path(raw).expanduser()
    try:
        brief = ReviewRequest.model_validate_json(
            path.read_text(encoding="utf-8"), strict=True
        )
    except OSError as exc:
        console.print(
            f"Cannot read {sanitize_terminal_text(str(path))}: {exc.strerror}"
        )
        return None
    except ValidationError as exc:
        console.print(
            Panel(
                "\n".join(validation_details(exc)),
                title="Invalid brief",
                border_style="red",
            )
        )
        return None
    return brief


def _validate_brief(console: Console, draft: dict[str, Any]) -> ReviewRequest | None:
    """Validate the draft brief, reporting every rejected field."""
    try:
        return ReviewRequest.model_validate(draft, strict=True)
    except ValidationError as exc:
        console.print(
            Panel(
                "\n".join(validation_details(exc)),
                title="Brief is not complete",
                border_style="red",
            )
        )
        return None


def _brief_panel(draft: dict[str, Any]) -> Panel:
    """Render the current brief draft with required-field markers."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    for field in (name for name, _ in BRIEF_FIELDS):
        marker = " *" if field in _REQUIRED else ""
        value = draft.get(field)
        rendered = "; ".join(value) if isinstance(value, list) else (value or "-")
        table.add_row(f"{field}{marker}", sanitize_terminal_text(rendered or "-"))
    return Panel(table, title="Application brief (* required)", border_style="blue")


def _edit_field(console: Console, field: str, current: Any) -> Any:
    """Read one field, keeping the current value when input is refused."""
    current_list = current if isinstance(current, list) else []
    if field in _LIST_FIELDS:
        joined = ", ".join(current_list)
        raw = Prompt.ask(field, default=joined, console=console)
        return [entry.strip() for entry in raw.split(",") if entry.strip()]
    current_text = current if isinstance(current, str) else ""
    console.print(
        f"{field}: enter text, '.end' to finish, or '.cancel' to keep the "
        "current value. A leading backslash keeps '.end' or '.cancel' as text."
    )
    lines: list[str] = []
    while True:
        line = console.input("")
        if line.startswith("\\") and line.strip() in {".end", ".cancel"}:
            lines.append(line[1:])
            continue
        if line.strip() == ".cancel":
            return None
        if line.strip() == ".end":
            break
        lines.append(line)
    entered = "\n".join(lines).strip()
    return entered or current_text


def _evidence_menu(
    console: Console, session: WorkspaceSession, draft: ReviewDraft
) -> bool:
    """Offer inspection, tracing, generation, and re-retrieval.

    Returns:
        (bool) Whether the caller should retrieve evidence again.
    """
    while True:
        choice = Prompt.ask(
            "Evidence",
            choices=["inspect", "trace", "generate", "again", "return"],
            default="generate",
            console=console,
        )
        if choice == "return":
            return False
        if choice == "again":
            return True
        if choice == "inspect":
            _inspect_item(console, session, draft)
        elif choice == "trace":
            console.print(render_retrieval_trace(draft.run.trace))
        elif choice == "generate":
            _generate(console, session, draft)


def _inspect_item(
    console: Console, session: WorkspaceSession, draft: ReviewDraft
) -> None:
    """Show one evidence item, and optionally its generated retrieval note."""
    index = Prompt.ask(
        "Evidence order number (1-based), or 'back'",
        default="back",
        console=console,
    )
    if index == "back":
        return
    try:
        position = int(index) - 1
        item = draft.run.selection.packet.items[position]
    except (ValueError, IndexError):
        console.print("No evidence item at that position.")
        return
    console.print(render_evidence_detail(item))
    if item.passage_id is None or not _confirmed(
        console, "Inspect the generated retrieval context?"
    ):
        return
    try:
        stored = session.generated_context(
            draft.run.selection.packet.generation_id, item.passage_id
        )
    except AtlasReviewError as error:
        print_error(console, error)
        return
    console.print(render_generated_context(stored.passage, stored.entry))


def _generate(console: Console, session: WorkspaceSession, draft: ReviewDraft) -> None:
    """Send one explicit request per action, then store or report it."""
    packet = draft.run.selection.packet
    while True:
        try:
            request = session.prepare_review_request(draft)
        except AtlasReviewError as error:
            print_error(console, error)
            return
        console.print(
            render_request_summary(session.config.llm, len(request.packet.items))
        )
        if len(request.packet.items) != len(packet.items):
            console.print(
                "Evidence trimmed to fit the request budget: "
                f"{len(request.packet.items)} of {len(packet.items)} items remain."
            )
            console.print(render_evidence_packet(request.packet))
        if not _confirmed(console, "Send this request?"):
            return
        try:
            outcome = session.generate_report(draft)
        except AtlasReviewError as error:
            print_error(console, error)
            if not _recover(console, session):
                return
            continue
        stored = session.open_review(outcome.review_id)
        console.print(render_review(stored, session.review_attempts(outcome.review_id)))
        if _confirmed(console, "Export this report as Markdown?"):
            _export(console, session, outcome.review_id)
        return


def _recover(console: Console, session: WorkspaceSession) -> bool:
    """Offer the documented recovery actions after a failed attempt."""
    action = Prompt.ask(
        "Action",
        choices=["retry", "configure", "return"],
        default="retry",
        console=console,
    )
    if action == "return":
        return False
    if action == "configure":
        edit_configuration(console, session)
    return True


def _export(console: Console, session: WorkspaceSession, review_id: str) -> None:
    """Export one saved review into the workspace exports directory."""
    try:
        target = session.export_review(review_id)
    except AtlasReviewError as error:
        print_error(console, error)
        return
    console.print(f"Exported {sanitize_terminal_text(str(target))}")


def _confirmed(console: Console, question: str) -> bool:
    """Ask a yes/no question with the documented default."""
    answer = Prompt.ask(question, choices=["y", "n"], default="n", console=console)
    return answer == "y"
