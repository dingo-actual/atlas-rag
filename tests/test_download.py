"""Bounded source download: repository pointer, redirects, limits, and service use.

These tests use a loopback HTTP server so the real ``urllib`` redirect and
timeout behaviour is exercised without reaching the internet.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from atlas_review.config import AppConfig
from atlas_review.corpus import download
from atlas_review.corpus.download import (
    LATEST_POINTER_URL,
    download_latest_source,
    download_source,
    is_permitted_source_url,
)
from atlas_review.errors import AtlasReviewError
from atlas_review.services.session import WorkspaceSession

pytestmark = pytest.mark.enable_socket

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_atlas_v6.yaml"
POINTER_PATH = "/dist/v6/ATLAS-latest.yaml"
RELEASE = "ATLAS-2026.10.yaml"
PATCH_RELEASE = "ATLAS-2022.03.2.yaml"


@dataclass(frozen=True, slots=True)
class Route:
    """One canned HTTP response.

    Instance Attributes:
        body (bytes): Response body.
        status (int): HTTP status code.
        location (str | None): Redirect target for 3xx responses.
        delay_seconds (float): Delay before the response is written.
    """

    body: bytes = b""
    status: int = 200
    location: str | None = None
    delay_seconds: float = 0.0


@dataclass(slots=True)
class LocalRepository:
    """A loopback repository that serves canned routes.

    Instance Attributes:
        root (str): Base URL of the server.
        routes (dict[str, Route]): Responses keyed by request path.
    """

    root: str
    routes: dict[str, Route] = field(default_factory=dict)

    def url(self, path: str) -> str:
        """Return the absolute URL for one served path."""
        return f"{self.root}{path}"

    def serve(self, path: str, route: Route) -> str:
        """Register one response and return its absolute URL."""
        self.routes[path] = route
        return self.url(path)


@pytest.fixture
def repository() -> Iterator[LocalRepository]:
    """Serve canned responses from a loopback HTTP server."""
    router = LocalRepository(root="")
    routes: dict[str, Route] = router.routes

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = routes.get(self.path)
            if route is None:
                self.send_error(404, "not found")
                return
            if route.delay_seconds:
                time.sleep(route.delay_seconds)
            self.send_response(route.status)
            if route.location is not None:
                self.send_header("Location", route.location)
            self.send_header("Content-Length", str(len(route.body)))
            self.end_headers()
            self.wfile.write(route.body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output free of request logs."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    router.root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield router
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("url", "permitted"),
    [
        pytest.param(
            "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/x.yaml",
            True,
            id="https-remote",
        ),
        pytest.param("http://127.0.0.1:8080/x.yaml", True, id="http-loopback-ip"),
        pytest.param("http://localhost:8080/x.yaml", True, id="http-localhost"),
        pytest.param("http://example.com/x.yaml", False, id="http-remote"),
        pytest.param("file:///etc/passwd", False, id="file"),
        pytest.param("ftp://example.com/x.yaml", False, id="ftp"),
        pytest.param("https:///x.yaml", False, id="no-host"),
    ],
)
def test_only_https_or_loopback_http_is_permitted(url: str, permitted: bool) -> None:
    """Refuse non-HTTP schemes and plain HTTP to a remote host."""
    assert is_permitted_source_url(url) is permitted


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(RELEASE, id="release"),
        pytest.param(PATCH_RELEASE, id="patch-release"),
        pytest.param(f"{RELEASE}\n", id="trailing-newline"),
        pytest.param(f"  {RELEASE}  ", id="surrounding-space"),
    ],
)
def test_latest_pointer_accepts_a_versioned_yaml_name(
    repository: LocalRepository, name: str
) -> None:
    """Follow a well-formed pointer to the release file it names."""
    body = b"format-version: 6.0.0\n"
    repository.serve(POINTER_PATH, Route(body=name.encode()))
    repository.serve(f"/dist/v6/{name.strip()}", Route(body=body))
    original = download.LATEST_POINTER_URL
    download.LATEST_POINTER_URL = repository.url(POINTER_PATH)
    try:
        result = download_latest_source(timeout_seconds=5.0, max_bytes=1024)
    finally:
        download.LATEST_POINTER_URL = original
    assert result.raw == body
    assert result.origin == repository.url(f"/dist/v6/{name.strip()}")
    assert result.pointer_value == name.strip()


@pytest.mark.parametrize(
    ("content", "code"),
    [
        pytest.param(b"", "unexpected_latest_source", id="empty"),
        pytest.param(b"../../etc/passwd", "unexpected_latest_source", id="traversal"),
        pytest.param(b"ATLAS-latest.yaml", "unexpected_latest_source", id="alias"),
        pytest.param(b"ATLAS-2026.10.yml", "unexpected_latest_source", id="extension"),
        pytest.param(
            b"https://example.com/ATLAS-2026.10.yaml",
            "unexpected_latest_source",
            id="absolute-url",
        ),
        pytest.param(b"A" * 2048, "source_too_large", id="oversized-pointer"),
    ],
)
def test_latest_pointer_must_name_one_versioned_file(
    repository: LocalRepository, content: bytes, code: str
) -> None:
    """Reject a pointer that is not exactly one ATLAS file name."""
    repository.serve(POINTER_PATH, Route(body=content))
    original = download.LATEST_POINTER_URL
    download.LATEST_POINTER_URL = repository.url(POINTER_PATH)
    try:
        with pytest.raises(AtlasReviewError) as caught:
            download_latest_source(timeout_seconds=5.0, max_bytes=1024)
    finally:
        download.LATEST_POINTER_URL = original
    assert caught.value.code == code
    assert caught.value.stage == "source_acquisition"


def test_latest_pointer_is_read_from_the_official_repository() -> None:
    """The default pointer stays inside the documented v6 directory."""
    assert LATEST_POINTER_URL.startswith("https://raw.githubusercontent.com/")
    assert LATEST_POINTER_URL.endswith(
        "/mitre-atlas/atlas-data/main/dist/v6/ATLAS-latest.yaml"
    )


def test_download_returns_requested_bytes(repository: LocalRepository) -> None:
    """A plain HTTPS-style download returns exact bytes and the source URL."""
    raw = FIXTURE.read_bytes()
    url = repository.serve("/source.yaml", Route(body=raw))
    result = download_source(url, timeout_seconds=5.0, max_bytes=len(raw))
    assert result.raw == raw
    assert result.origin == url


def test_download_follows_a_same_host_redirect(repository: LocalRepository) -> None:
    """A redirect to another path on the same loopback host is followed."""
    raw = b"format-version: 6.0.0\n"
    url = repository.serve("/moving.yaml", Route(status=302, location="/source.yaml"))
    repository.serve("/source.yaml", Route(body=raw))
    assert download_source(url, timeout_seconds=5.0, max_bytes=1024).raw == raw


def test_download_refuses_a_redirect_to_a_file_scheme(
    repository: LocalRepository,
) -> None:
    """A redirect must never leave HTTP for a local file."""
    url = repository.serve(
        "/moving.yaml", Route(status=302, location="file:///etc/passwd")
    )
    with pytest.raises(AtlasReviewError) as caught:
        download_source(url, timeout_seconds=5.0, max_bytes=1024)
    assert caught.value.code == "untrusted_source_url"
    assert caught.value.stage == "source_acquisition"


@pytest.mark.parametrize("status", [403, 404, 500])
def test_download_reports_an_http_error(
    repository: LocalRepository, status: int
) -> None:
    """An HTTP error becomes an acquisition error, not a traceback."""
    url = repository.serve("/missing.yaml", Route(status=status))
    with pytest.raises(AtlasReviewError) as caught:
        download_source(url, timeout_seconds=5.0, max_bytes=1024)
    assert caught.value.code == "source_download_failed"
    assert str(status) in caught.value.message


def test_download_refuses_a_body_above_the_limit(
    repository: LocalRepository,
) -> None:
    """The byte limit is enforced while reading, not only from the header."""
    url = repository.serve("/big.yaml", Route(body=b"x" * 65))
    with pytest.raises(AtlasReviewError) as caught:
        download_source(url, timeout_seconds=5.0, max_bytes=64)
    assert caught.value.code == "source_too_large"


def test_download_times_out(repository: LocalRepository) -> None:
    """A slow server is abandoned with a timeout error."""
    url = repository.serve("/slow.yaml", Route(body=b"x", delay_seconds=1.0))
    with pytest.raises(AtlasReviewError) as caught:
        download_source(url, timeout_seconds=0.2, max_bytes=1024)
    assert caught.value.code == "source_download_timeout"


def test_download_refuses_an_empty_body(repository: LocalRepository) -> None:
    """A zero-byte source is not a usable corpus."""
    url = repository.serve("/empty.yaml", Route(body=b""))
    with pytest.raises(AtlasReviewError) as caught:
        download_source(url, timeout_seconds=5.0, max_bytes=1024)
    assert caught.value.code == "empty_source"


def test_prepare_latest_source_plans_the_downloaded_bytes(
    fake_session: WorkspaceSession, repository: LocalRepository
) -> None:
    """The latest pointer yields a preview, plan, and bytes before activation."""
    raw = FIXTURE.read_bytes()
    repository.serve(POINTER_PATH, Route(body=RELEASE.encode()))
    file_url = repository.serve(f"/dist/v6/{RELEASE}", Route(body=raw))
    original = download.LATEST_POINTER_URL
    download.LATEST_POINTER_URL = repository.url(POINTER_PATH)
    try:
        prepared = fake_session.prepare_latest_source()
    finally:
        download.LATEST_POINTER_URL = original
    assert prepared.source_bytes == raw
    assert prepared.preview.origin == file_url
    assert prepared.preview.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert prepared.preview.acquired_at
    assert prepared.plan.chunk_count > 0
    assert fake_session.generations() == ()


def test_prepare_default_source_uses_the_configured_url(
    fake_session: WorkspaceSession, repository: LocalRepository
) -> None:
    """The documented default source is the configured corpus URL."""
    raw = FIXTURE.read_bytes()
    file_url = repository.serve("/dist/v6/ATLAS-2026.09.yaml", Route(body=raw))
    config = AppConfig.model_validate(
        {
            **fake_session.config.model_dump(mode="json"),
            "corpus": {
                **fake_session.config.corpus.model_dump(mode="json"),
                "default_url": file_url,
            },
        },
        strict=True,
    )
    fake_session.save_config(config)
    prepared = fake_session.prepare_default_source()
    assert prepared.preview.origin == file_url
    assert prepared.source_bytes == raw


def test_downloaded_bytes_keep_the_local_source_identity(
    fake_session: WorkspaceSession, source_path: Path, repository: LocalRepository
) -> None:
    """The same bytes produce the same digest regardless of origin."""
    raw = source_path.read_bytes()
    file_url = repository.serve("/dist/v6/ATLAS-2026.09.yaml", Route(body=raw))
    downloaded = download_source(file_url, timeout_seconds=5.0, max_bytes=len(raw))
    local = fake_session.preview_file(source_path)
    assert hashlib.sha256(downloaded.raw).hexdigest() == local.source_sha256


def test_unsupported_upstream_schema_is_reported_after_download(
    fake_session: WorkspaceSession, source_bytes: bytes, repository: LocalRepository
) -> None:
    """An unknown relationship type fails with the parser's actionable error."""
    unsupported = source_bytes.replace(b"employs:", b"owns:", 1)
    assert unsupported != source_bytes
    repository.serve(POINTER_PATH, Route(body=RELEASE.encode()))
    repository.serve(f"/dist/v6/{RELEASE}", Route(body=unsupported))
    original = download.LATEST_POINTER_URL
    download.LATEST_POINTER_URL = repository.url(POINTER_PATH)
    try:
        with pytest.raises(AtlasReviewError) as caught:
            fake_session.prepare_latest_source()
    finally:
        download.LATEST_POINTER_URL = original
    assert caught.value.code == "unsupported_relationship_type"
    assert fake_session.generations() == ()
