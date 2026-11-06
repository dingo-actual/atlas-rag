"""Command-line entry point for the ATLAS security review workspace."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession
from atlas_review.tui.app import print_error, run_tui
from atlas_review.tui.render import render_preview

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
