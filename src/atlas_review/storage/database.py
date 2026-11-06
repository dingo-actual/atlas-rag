"""Workspace SQLite connection policy, schema checks, and transactions."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self

from atlas_review.errors import STAGE_DATABASE, AtlasReviewError
from atlas_review.storage.schema import SCHEMA_SQL, SCHEMA_VERSION

BUSY_TIMEOUT_MS = 5000


class WorkspaceDatabase:
    """Own one SQLite connection bound to a workspace database file.

    Instance Attributes:
        path (Path): Workspace database file.

    Example Usage:
        ```python
            >>> with WorkspaceDatabase(workspace / "atlas.sqlite3") as db:
            ...     with db.transaction() as connection:
            ...         connection.execute("SELECT 1")
        ```
    """

    def __init__(self, path: Path) -> None:
        """Store the database path without opening a connection.

        Args:
            path (Path): Workspace database file.
        """
        self._path = path
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        """Return the workspace database file."""
        return self._path

    def connect(self) -> sqlite3.Connection:
        """Open the connection if needed and enforce the schema version.

        Returns:
            (sqlite3.Connection) Open connection with foreign keys enabled.

        Raises:
            AtlasReviewError: The file is unavailable or carries an
                unsupported schema version.
        """
        if self._connection is not None:
            return self._connection
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path, isolation_level=None)
        except (OSError, sqlite3.Error) as exc:
            raise _database_error(
                "unavailable_database",
                f"cannot open workspace database {self._path}",
            ) from exc
        connection.row_factory = sqlite3.Row
        self._connection = connection
        try:
            _apply_pragmas(connection)
            self._ensure_schema(connection)
        except (AtlasReviewError, sqlite3.Error):
            self.close()
            raise
        return connection

    def close(self) -> None:
        """Close the connection if it is open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Run one explicit immediate transaction, rolling back on failure.

        Yields:
            (sqlite3.Connection) Connection inside the transaction.

        Raises:
            sqlite3.Error: The transaction could not begin or commit.
        """
        connection = self.connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    def schema_version(self) -> int:
        """Return the ``PRAGMA user_version`` value of the open workspace."""
        row = self.connect().execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        """Create an empty workspace schema or refuse an unsupported one."""
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == SCHEMA_VERSION:
            return
        if version == 0 and not _has_tables(connection):
            connection.executescript(SCHEMA_SQL)
            return
        raise _database_error(
            "unsupported_database_schema",
            f"workspace schema version {version} is not supported "
            f"(expected {SCHEMA_VERSION})",
            "Create a new workspace or migrate the database explicitly.",
        )


def _apply_pragmas(connection: sqlite3.Connection) -> None:
    """Enable foreign keys and a finite lock wait for this connection."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")


def _has_tables(connection: sqlite3.Connection) -> bool:
    """Report whether the database already contains application tables."""
    row = connection.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    return int(row[0]) > 0


def _database_error(code: str, message: str, recovery: str = "") -> AtlasReviewError:
    """Build a database-stage error with a workspace recovery hint."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_DATABASE,
        message=message,
        recovery=recovery
        or "Choose a writable workspace directory or create a new workspace.",
    )
