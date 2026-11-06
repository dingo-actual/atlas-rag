"""Rich main menu for source ingestion and corpus inspection."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.tui.render import (
    render_generations,
    render_inspection,
    render_preview,
)

INSPECTABLE_KINDS = ("technique", "mitigation", "case_study")


def run_tui(console: Console, session: WorkspaceSession) -> None:
    """Run the main menu until the operator quits.

    Args:
        console (Console): Rich console supplied by the entry point.
        session (WorkspaceSession): Open workspace session.
    """
    while True:
        console.print(render_generations(session.generations()))
        action = Prompt.ask(
            "Action",
            choices=["ingest", "inspect", "refresh", "quit"],
            default="inspect",
            console=console,
        )
        if action == "quit":
            return
        try:
            _dispatch(console, session, action)
        except AtlasReviewError as error:
            print_error(console, error)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return


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


def _dispatch(console: Console, session: WorkspaceSession, action: str) -> None:
    """Run one menu action."""
    if action == "ingest":
        _ingest(console, session)
    elif action == "inspect":
        _inspect(console, session)


def _ingest(console: Console, session: WorkspaceSession) -> None:
    """Prompt for a source file, preview it, and store it on confirmation."""
    answer = Prompt.ask("Source file", console=console)
    path = Path(answer)
    console.print(render_preview(session.preview_file(path)))
    if (
        Prompt.ask(
            "Store this generation?",
            choices=["yes", "no"],
            default="no",
            console=console,
        )
        == "no"
    ):
        console.print("Ingestion cancelled.", style="yellow")
        return
    outcome = session.ingest_file(path)
    console.print(f"stored generation {outcome.generation_id}")


def _inspect(console: Console, session: WorkspaceSession) -> None:
    """Prompt for an entity and print its inspection view."""
    kind = Prompt.ask(
        "Kind",
        choices=list(INSPECTABLE_KINDS),
        default="technique",
        console=console,
    )
    choices = session.entity_choices(kind)
    if not choices:
        console.print(f"No stored {kind} records.", style="yellow")
        return
    for entity_id, display_name in choices:
        console.print(f"  {entity_id}  {display_name}")
    entity_id = Prompt.ask(
        "Entity id",
        choices=[entity_id for entity_id, _ in choices],
        console=console,
    )
    console.print(render_inspection(session.inspect(entity_id)))
