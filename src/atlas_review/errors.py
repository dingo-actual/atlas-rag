"""Application error taxonomy with stable codes, stages, and recovery hints."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

STAGE_CONFIGURATION = "configuration"
STAGE_SOURCE_ACQUISITION = "source_acquisition"
STAGE_SOURCE_SCHEMA = "source_schema"
STAGE_STORAGE = "storage"
STAGE_DATABASE = "database"
STAGE_INGESTION = "ingestion"
STAGE_UI = "ui"


@dataclass(frozen=True, slots=True)
class AtlasReviewError(Exception):
    """Expected application failure carrying a safe message and recovery action.

    Args:
        code (str): Stable machine-readable identifier.
        stage (str): Pipeline stage that rejected the work.
        message (str): Short, sanitized message safe to display.
        details (tuple[str, ...]): Optional sanitized detail lines.
        recovery (str): Suggested next action for the user.

    Example Usage:
        ```python
            >>> raise AtlasReviewError(
            ...     code="unsupported_source_format",
            ...     stage=STAGE_SOURCE_SCHEMA,
            ...     message="unsupported ATLAS format major 5",
            ...     recovery="Select a format 6 source file.",
            ... )
        ```
    """

    code: str
    stage: str
    message: str
    details: tuple[str, ...] = field(default_factory=tuple)
    recovery: str = ""

    def __str__(self) -> str:
        """Render the message and stable code."""
        return f"{self.message} ({self.code})"


def validation_details(error: ValidationError) -> tuple[str, ...]:
    """Return field paths and error categories without any input values.

    Args:
        error (ValidationError): Pydantic failure raised by a draft payload.

    Returns:
        (tuple[str, ...]) One ``field.path: category`` line per error.
    """
    return tuple(
        f"{'.'.join(str(part) for part in item['loc']) or '<payload>'}: {item['type']}"
        for item in error.errors(include_input=False, include_url=False)
    )
