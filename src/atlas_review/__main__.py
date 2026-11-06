"""Module form of the command-line entry point."""

from __future__ import annotations

import sys

from atlas_review.cli import main

if __name__ == "__main__":
    sys.exit(main())
