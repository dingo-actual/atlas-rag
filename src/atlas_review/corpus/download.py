"""Bounded download of the pinned and latest ATLAS v6 sources.

The ATLAS repository publishes one versioned YAML file per collection under
``dist/v6/`` and keeps ``dist/v6/ATLAS-latest.yaml`` naming the newest one.
Reading that pointer avoids a directory listing and its rate limits; the name
it contains must match one versioned file pattern exactly.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from atlas_review.errors import STAGE_SOURCE_ACQUISITION, AtlasReviewError

ATLAS_REPOSITORY = "https://github.com/mitre-atlas/atlas-data"
LATEST_POINTER_URL = (
    "https://raw.githubusercontent.com/mitre-atlas/atlas-data/"
    "main/dist/v6/ATLAS-latest.yaml"
)
MAX_POINTER_BYTES = 1024
_USER_AGENT = "atlas-review"
_LATEST_FILE = re.compile(r"ATLAS-\d{4}\.\d{2}(?:\.\d+)?\.yaml")
_MAX_DETAIL_CHARS = 120


@dataclass(frozen=True, slots=True)
class DownloadedSource:
    """Exact source bytes and where the download came from.

    Instance Attributes:
        raw (bytes): Downloaded source bytes.
        origin (str): Absolute URL of the downloaded document.
        pointer_url (str | None): Pointer URL, when the latest source was read.
        pointer_value (str | None): Pointer text naming the document.
    """

    raw: bytes
    origin: str
    pointer_url: str | None = None
    pointer_value: str | None = None


def is_permitted_source_url(url: str) -> bool:
    """Report whether a source URL uses HTTPS or loopback HTTP.

    Args:
        url (str): Candidate absolute URL.

    Returns:
        (bool) ``True`` for HTTPS with a host, and for loopback HTTP.
    """
    parts = urlsplit(url)
    if not parts.hostname:
        return False
    if parts.scheme == "https":
        return True
    if parts.scheme != "http":
        return False
    if parts.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        return False


def download_source(
    url: str, *, timeout_seconds: float, max_bytes: int
) -> DownloadedSource:
    """Download one permitted source URL within the configured bounds.

    Args:
        url (str): Absolute HTTPS or loopback HTTP URL.
        timeout_seconds (float): Finite connection and read timeout.
        max_bytes (int): Largest accepted response body.

    Returns:
        (DownloadedSource) Exact bytes and the URL they came from.

    Raises:
        AtlasReviewError: The URL is refused, unreachable, slow, empty, or
            larger than the limit.
    """
    raw = _read_url(url, limit=max_bytes, timeout_seconds=timeout_seconds)
    if not raw:
        raise _acquisition_error("empty_source", f"the source at {url} is empty")
    return DownloadedSource(raw=raw, origin=url)


def download_latest_source(
    *, timeout_seconds: float, max_bytes: int
) -> DownloadedSource:
    """Follow the ATLAS v6 latest pointer and download the file it names.

    Args:
        timeout_seconds (float): Finite timeout for each request.
        max_bytes (int): Largest accepted document body.

    Returns:
        (DownloadedSource) Bytes, document URL, and pointer details.

    Raises:
        AtlasReviewError: The pointer is missing or malformed, or the document
            download fails or exceeds the limit.
    """
    pointer_raw = _read_url(
        LATEST_POINTER_URL,
        limit=MAX_POINTER_BYTES,
        timeout_seconds=timeout_seconds,
    )
    pointer_value = _pointer_name(pointer_raw)
    origin = urljoin(LATEST_POINTER_URL, pointer_value)
    raw = _read_url(origin, limit=max_bytes, timeout_seconds=timeout_seconds)
    if not raw:
        raise _acquisition_error(
            "empty_source", f"the latest source {pointer_value} is empty"
        )
    return DownloadedSource(
        raw=raw,
        origin=origin,
        pointer_url=LATEST_POINTER_URL,
        pointer_value=pointer_value,
    )


def _pointer_name(raw: bytes) -> str:
    """Return the one ATLAS file name the latest pointer contains."""
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        value = ""
    if not _LATEST_FILE.fullmatch(value):
        raise _acquisition_error(
            "unexpected_latest_source",
            "the latest-source pointer does not name an ATLAS v6 YAML file",
            (f"pointer value: {value[:_MAX_DETAIL_CHARS]!r}",),
        )
    return value


def _read_url(url: str, *, limit: int, timeout_seconds: float) -> bytes:
    """Read one URL through a redirect policy that only permits safe schemes."""
    _require_permitted(url)
    opener = build_opener(_RedirectPolicy())
    request = Request(  # noqa: S310 - _require_permitted limits the scheme
        url, headers={"User-Agent": _USER_AGENT}
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            _require_permitted(response.geturl())
            declared = response.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > limit:
                raise _too_large(limit)
            raw = cast("bytes", response.read(limit + 1))
    except HTTPError as exc:
        raise _acquisition_error(
            "source_download_failed",
            f"the source server returned HTTP {exc.code}",
        ) from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise _acquisition_error(
                "source_download_timeout", "the source download timed out"
            ) from exc
        raise _acquisition_error(
            "source_download_failed", "the source server could not be reached"
        ) from exc
    except TimeoutError as exc:
        raise _acquisition_error(
            "source_download_timeout", "the source download timed out"
        ) from exc
    except OSError as exc:
        raise _acquisition_error(
            "source_download_failed", "the source download failed"
        ) from exc
    if len(raw) > limit:
        raise _too_large(limit)
    return raw


class _RedirectPolicy(HTTPRedirectHandler):
    """Follow redirects only to HTTPS or loopback HTTP."""

    def http_error_302(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        """Refuse a redirect location that leaves the permitted schemes."""
        location = headers.get("location") or headers.get("uri")
        if location is not None:
            _require_permitted(urljoin(req.full_url, location))
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        """Refuse a redirect request that leaves the permitted schemes."""
        _require_permitted(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _require_permitted(url: str) -> None:
    """Refuse a source URL that is neither HTTPS nor loopback HTTP."""
    if not is_permitted_source_url(url):
        scheme = urlsplit(url).scheme
        raise _acquisition_error(
            "untrusted_source_url",
            "the source URL must use HTTPS or a loopback HTTP server",
            (f"scheme: {scheme[:_MAX_DETAIL_CHARS] or 'none'}",),
        )


def _too_large(limit: int) -> AtlasReviewError:
    """Build the oversized-source error for a byte limit."""
    return _acquisition_error(
        "source_too_large", f"the source exceeds the {limit}-byte limit"
    )


def _acquisition_error(
    code: str, message: str, details: tuple[str, ...] = ()
) -> AtlasReviewError:
    """Build a source-acquisition error with a download recovery hint."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_SOURCE_ACQUISITION,
        message=message,
        details=details,
        recovery=(
            "Check network access and the ATLAS repository, or load a local "
            f"v6 YAML file from {ATLAS_REPOSITORY}."
        ),
    )
