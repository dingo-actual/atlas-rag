"""Rich main menu for explicit ingestion, review, inspection, and configuration."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.storage.reviews import ReviewSummary
from atlas_review.tui.configuration import edit_configuration
from atlas_review.tui.render import (
    inference_label,
    print_error,
    render_build_outcome,
    render_generations,
    render_ingest_plan,
    render_inspection,
    render_menu,
    render_preview,
    render_status,
    sanitize_terminal_text,
)
from atlas_review.tui.review import run_review
from atlas_review.tui.review_render import (
    render_request_summary,
    render_review,
    render_reviews,
)

__all__ = ["run_tui"]

MENU_CHOICES: tuple[tuple[str, str], ...] = (
    ("1", "Ingest ATLAS data"),
    ("2", "Run inference / security review"),
    ("3", "Configure inference"),
    ("4", "Inspect corpus and saved reviews"),
    ("0", "Exit"),
)


def run_tui(console: Console, session: WorkspaceSession) -> None:
    """Run the main menu until the user exits.

    Args:
        console (Console): Rich console.
        session (WorkspaceSession): Open workspace session.
    """
    while True:
        console.print(_status(session))
        console.print(render_generations(session.generations()))
        console.print(render_menu(MENU_CHOICES))
        choice = Prompt.ask(
            "Choice",
            choices=[key for key, _ in MENU_CHOICES],
            default="2",
            console=console,
        )
        try:
            if not _dispatch(console, session, choice):
                return
        except AtlasReviewError as error:
            print_error(console, error)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return


def _dispatch(console: Console, session: WorkspaceSession, choice: str) -> bool:
    """Run one menu action and report whether the menu should continue."""
    if choice == "0":
        return False
    if choice == "1":
        _ingest(console, session)
    elif choice == "2":
        run_review(console, session)
    elif choice == "3":
        edit_configuration(console, session)
    else:
        _inspect(console, session)
    return True


def _status(session: WorkspaceSession) -> object:
    """Render the header for the current workspace state."""
    active = session.repository.active_generation_id()
    collection = next(
        (
            summary.collection_version
            for summary in session.generations()
            if summary.generation_id == active
        ),
        "no active generation",
    )
    return render_status(
        workspace=session.paths.root,
        collection=collection,
        inference=inference_label(session.config.llm),
    )


def _ingest(console: Console, session: WorkspaceSession) -> None:
    """Ask for a source, then run explicit contextualized ingestion."""
    console.print(
        "default: pinned documented source | latest: newest ATLAS 6.x release | "
        "file: local YAML",
        style="dim",
    )
    source = Prompt.ask(
        "Source",
        choices=["default", "latest", "file"],
        default="default",
        console=console,
    )
    if source == "default":
        prepared = session.prepare_default_source()
    elif source == "latest":
        prepared = session.prepare_latest_source()
    else:
        raw = Prompt.ask("ATLAS v6 YAML path", default="", console=console).strip()
        if not raw:
            return
        prepared = session.prepare_ingestion(Path(raw).expanduser())
    console.print(render_preview(prepared.preview))
    console.print(render_ingest_plan(prepared.plan))
    answer = Prompt.ask(
        "Build and activate this context now?",
        choices=["y", "n"],
        default="n",
        console=console,
    )
    if answer != "y":
        return
    console.print(render_build_outcome(session.build_generation(prepared)))


def _inspect(console: Console, session: WorkspaceSession) -> None:
    """Inspect the stored corpus or reopen and export saved reviews."""
    choice = Prompt.ask(
        "Inspect",
        choices=["corpus", "reviews", "return"],
        default="reviews",
        console=console,
    )
    if choice == "corpus":
        entity_id = Prompt.ask(
            "ATLAS entity ID, or 'back'", default="back", console=console
        )
        if entity_id != "back":
            console.print(render_inspection(session.inspect(entity_id)))
    elif choice == "reviews":
        _reviews(console, session)


def _reviews(console: Console, session: WorkspaceSession) -> None:
    """List saved reviews, then reopen or export one without a new request."""
    summaries = session.saved_reviews()
    console.print(render_reviews(summaries))
    if not summaries:
        return
    selected = Prompt.ask("Review ID, or 'back'", default="back", console=console)
    match = _match(summaries, selected)
    if match is None:
        return
    console.print(
        render_review(
            session.open_review(match.review_id),
            session.review_attempts(match.review_id),
        )
    )
    action = Prompt.ask(
        "Action",
        choices=["export", "regenerate", "return"],
        default="return",
        console=console,
    )
    if action == "export":
        console.print(
            f"Exported {sanitize_terminal_text(str(session.export_review(match.review_id)))}"
        )
    elif action == "regenerate":
        _regenerate(console, session, match.review_id)


def _regenerate(console: Console, session: WorkspaceSession, review_id: str) -> None:
    """Send a new attempt for a reopened review with its frozen evidence."""
    stored = session.open_review(review_id)
    console.print(
        render_request_summary(session.config.llm, len(stored.evidence.items))
    )
    answer = Prompt.ask(
        "Send a new request with the same stored evidence?",
        choices=["y", "n"],
        default="n",
        console=console,
    )
    if answer != "y":
        return
    outcome = session.regenerate_review(review_id)
    console.print(
        render_review(
            session.open_review(outcome.review_id),
            session.review_attempts(outcome.review_id),
        )
    )


def _match(summaries: tuple[ReviewSummary, ...], selected: str) -> ReviewSummary | None:
    """Find a saved review by exact identifier or unique prefix."""
    if selected == "back":
        return None
    for summary in summaries:
        if summary.review_id == selected:
            return summary
    matches = [
        summary for summary in summaries if summary.review_id.startswith(selected)
    ]
    return matches[0] if len(matches) == 1 else None
