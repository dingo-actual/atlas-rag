"""Display-safe rendering of untrusted text.

ATLAS excerpts, application briefs, and model output are untrusted data. They
are stripped of terminal escape sequences and control characters before they
reach a terminal or an exported document.
"""

from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_ESCAPES = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def sanitize_text(value: str) -> str:
    """Strip control characters and escape sequences from untrusted text.

    Args:
        value (str): Text that may contain ATLAS or model content.

    Returns:
        (str) Text safe to display, with intentional newlines preserved.
    """
    return _CONTROL.sub("", _ESCAPES.sub("", value))
