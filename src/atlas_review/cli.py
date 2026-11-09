"""Command-line entry point for the ATLAS security review workspace."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from atlas_review.errors import STAGE_DATABASE, AtlasReviewError
from atlas_review.services.session import WorkspacePaths, WorkspaceSession
from atlas_review.storage.migrations import (
    PREVIOUS_SCHEMA_VERSION,
    migrate_workspace,
    stored_schema_version,
)
from atlas_review.storage.schema import SCHEMA_VERSION
from atlas_review.tui.app import run_tui
from atlas_review.tui.render import print_error, render_preview

DEFAULT_WORKSPACE = Path("./.atlas-review")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        (argparse.ArgumentParser) Parser with the workspace and source options.
    """
    parser = argparse.ArgumentParser(
        prog="atlas-review",
        description="Local ATLAS security review workspace.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="workspace directory (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="ingest one ATLAS v6 YAML file, print its preview, and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the workspace menu or a single ingestion.

    Args:
        argv (Sequence[str] | None): Arguments; ``None`` uses ``sys.argv``.

    Returns:
        (int) Process exit status.
    """
    arguments = build_parser().parse_args(argv)
    console = Console()
    session: WorkspaceSession | None = None
    try:
        upgrade_workspace(console, arguments.workspace)
        session = WorkspaceSession(arguments.workspace)
    except AtlasReviewError as error:
        print_error(console, error)
        return 1
    try:
        if arguments.source is not None:
            outcome = session.ingest_file(arguments.source)
            console.print(render_preview(outcome.preview))
            console.print(f"stored generation {outcome.generation_id}")
        else:
            run_tui(console, session)
    except AtlasReviewError as error:
        print_error(console, error)
        return 1
    else:
        return 0
    finally:
        session.close()


def upgrade_workspace(console: Console, workspace: Path) -> None:
    """Offer an explicit in-place schema migration before opening a workspace.

    Args:
        console (Console): Rich console for the notice and question.
        workspace (Path): Workspace root supplied on the command line.

    Raises:
        AtlasReviewError: The operator declines the migration.
    """
    database = WorkspacePaths.under(workspace).database
    version = stored_schema_version(database)
    if version != PREVIOUS_SCHEMA_VERSION:
        return
    console.print(
        f"This workspace uses schema version {version}; this release uses "
        f"version {SCHEMA_VERSION}.",
        style="yellow",
    )
    if not Confirm.ask(
        "Migrate the workspace in place now? Stored generations and reviews "
        "are preserved.",
        default=False,
        console=console,
    ):
        raise AtlasReviewError(
            code="workspace_migration_declined",
            stage=STAGE_DATABASE,
            message="the workspace schema is older than this release",
            recovery=(
                "Run the application again and accept the migration, or create "
                "a new workspace."
            ),
        )
    migrate_workspace(database)
    console.print("Workspace migrated; stored generations and reviews are unchanged.")
