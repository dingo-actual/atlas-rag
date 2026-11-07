"""Rich main menu for explicit ingestion, inspection, and retrieval."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.tui.render import (
    render_build_outcome,
    render_generations,
    render_ingest_plan,
    render_inspection,
    render_preview,
    render_retrieval_trace,
)

INSPECTABLE_KINDS = ("technique", "mitigation", "case_study")
ACTIONS = ("ingest", "search", "inspect", "refresh", "quit")


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
            choices=list(ACTIONS),
            default="search",
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
    elif action == "search":
        _search(console, session)
    elif action == "inspect":
        _inspect(console, session)


def _ingest(console: Console, session: WorkspaceSession) -> None:
    """Plan a source, show the plan, and build it on confirmation."""
    answer = Prompt.ask("Source file", console=console)
    console.print("loading retrieval tokenizers (first use downloads artifacts)")
    prepared = session.prepare_ingestion(Path(answer))
    console.print(render_preview(prepared.preview))
    console.print(render_ingest_plan(prepared.plan))
    if not _confirmed(console, "Build this generation?"):
        console.print("Ingestion cancelled.", style="yellow")
        return
    outcome = session.build_generation(prepared, progress=console.print)
    console.print(render_build_outcome(outcome))


def _search(console: Console, session: WorkspaceSession) -> None:
    """Ask for a retrieval focus and render the traced local run."""
    focus = Prompt.ask("Retrieval focus", console=console)
    if not focus.strip():
        console.print("Retrieval cancelled.", style="yellow")
        return
    console.print(
        "loading retrieval models and reranker (first use downloads artifacts)"
    )
    console.print(render_retrieval_trace(session.retrieve(focus).trace))


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


def _confirmed(console: Console, question: str) -> bool:
    """Ask one yes/no question defaulting to no."""
    answer = Prompt.ask(question, choices=["yes", "no"], default="no", console=console)
    return answer == "yes"
