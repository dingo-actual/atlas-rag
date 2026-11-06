"""Stored application configuration: load, initialise, and replace."""

from __future__ import annotations

from atlas_review.config import AppConfig
from atlas_review.errors import STAGE_STORAGE, AtlasReviewError
from atlas_review.storage.codec import decode_model, encode_model
from atlas_review.storage.database import WorkspaceDatabase


def load_config(database: WorkspaceDatabase, *, updated_at: str) -> AppConfig:
    """Return the stored configuration, initialising defaults on first use.

    Args:
        database (WorkspaceDatabase): Open workspace database owner.
        updated_at (str): ISO 8601 timestamp used when defaults are written.

    Returns:
        (AppConfig) Validated stored configuration.

    Raises:
        AtlasReviewError: The stored payload fails validation or the stored
            schema version does not match the model.
    """
    row = (
        database.connect()
        .execute(
            "SELECT schema_version, payload_msgpack FROM configuration "
            "WHERE singleton = 1"
        )
        .fetchone()
    )
    if row is None:
        config = AppConfig()
        save_config(database, config, updated_at=updated_at)
        return config
    if int(row["schema_version"]) != config_schema_version():
        raise AtlasReviewError(
            code="unsupported_configuration_schema",
            stage=STAGE_STORAGE,
            message=(
                f"stored configuration schema version {row['schema_version']} "
                f"is not supported"
            ),
            recovery="Export the configuration, then create a new workspace.",
        )
    return decode_model(row["payload_msgpack"], AppConfig)


def save_config(
    database: WorkspaceDatabase, config: AppConfig, *, updated_at: str
) -> None:
    """Validate and store the whole configuration in one row.

    Args:
        database (WorkspaceDatabase): Open workspace database owner.
        config (AppConfig): Configuration to store.
        updated_at (str): ISO 8601 timestamp for the update.
    """
    validated = AppConfig.model_validate(config, strict=True)
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO configuration (singleton, schema_version, updated_at,
                                       payload_msgpack)
            VALUES (1, ?, ?, ?)
            ON CONFLICT (singleton) DO UPDATE SET
                schema_version = excluded.schema_version,
                updated_at = excluded.updated_at,
                payload_msgpack = excluded.payload_msgpack
            """,
            (validated.schema_version, updated_at, encode_model(validated)),
        )


def config_schema_version() -> int:
    """Return the configuration schema version expected by this release."""
    return int(AppConfig.model_fields["schema_version"].default)
