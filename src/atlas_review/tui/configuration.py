"""Configuration screen: provider settings, import, and export."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from atlas_review.config import (
    AppConfig,
    LLMConfig,
    apply_provider_settings,
    config_from_json,
    config_to_json,
)
from atlas_review.errors import AtlasReviewError, validation_details
from atlas_review.services.session import WorkspaceSession
from atlas_review.tui.render import print_error, sanitize_terminal_text
from atlas_review.tui.review_render import render_configuration

_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("provider_name", "Provider name"),
    ("base_url", "API root URL"),
    ("model", "Model"),
    ("api_key_env", "Credential environment-variable name"),
    ("output_token_parameter", "Output token parameter"),
    ("response_format_mode", "Response format mode"),
)


def edit_configuration(console: Console, session: WorkspaceSession) -> None:
    """Show and edit the stored inference configuration.

    Args:
        console (Console): Rich console.
        session (WorkspaceSession): Open workspace session.
    """
    config = session.config.llm
    console.print(
        render_configuration(config, credential_present=_credential_present(config))
    )
    action = Prompt.ask(
        "Configuration",
        choices=[
            "edit",
            "import",
            "import-app",
            "export-provider",
            "export-app",
            "return",
        ],
        default="return",
        console=console,
    )
    if action == "edit":
        _edit(console, session)
    elif action == "import":
        _import(console, session)
    elif action == "import-app":
        _import_app(console, session)
    elif action == "export-provider":
        _write(
            console,
            session.paths.exports / "provider-settings.json",
            _provider_json(config),
        )
    elif action == "export-app":
        _write(
            console,
            session.paths.exports / "configuration.json",
            config_to_json(session.config),
        )


def _edit(console: Console, session: WorkspaceSession) -> None:
    """Prompt for every non-secret provider setting, then validate and store."""
    config = session.config.llm
    raw: dict[str, object] = {
        "provider_name": config.provider_name,
        "base_url": config.base_url or "",
        "model": config.model,
        "api_key_mode": config.api_key_mode,
        "api_key_env": config.api_key_env,
        "output_tokens": config.output_tokens,
        "output_token_parameter": config.output_token_parameter,
        "response_format_mode": config.response_format_mode,
        "max_input_characters": config.max_input_characters,
        "max_findings": config.max_findings,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
    }
    for field, label in _TEXT_FIELDS:
        raw[field] = Prompt.ask(
            label, default=str(raw[field] or ""), console=console
        ).strip()
    raw["api_key_mode"] = Prompt.ask(
        "API key mode",
        choices=["env", "none"],
        default=config.api_key_mode,
        console=console,
    )
    if raw["api_key_mode"] == "none":
        raw["api_key_env"] = None
    for field, label in (
        ("output_tokens", "Output tokens"),
        ("max_input_characters", "Request character budget"),
        ("max_findings", "Maximum findings"),
    ):
        raw[field] = _positive_int(console, label, raw[field])
    _apply(console, session, raw)


def _positive_int(console: Console, label: str, current: object) -> int | None:
    """Read a positive integer, or None when the input is not usable."""
    entered = Prompt.ask(label, default=str(current), console=console).strip()
    try:
        value = int(entered)
    except ValueError:
        console.print(f"{label} must be a whole number.")
        return None
    if value <= 0:
        console.print(f"{label} must be positive.")
        return None
    return value


def _apply(console: Console, session: WorkspaceSession, raw: dict[str, object]) -> None:
    """Validate the edited settings and store them when they are usable."""
    try:
        provider = LLMConfig.model_validate(
            {
                key: value
                for key, value in raw.items()
                if value is not None or key == "api_key_env"
            },
            strict=True,
        )
        config = AppConfig.model_validate(
            {
                **session.config.model_dump(mode="json"),
                "llm": provider.model_dump(mode="json"),
            },
            strict=True,
        )
        session.save_config(config)
    except ValidationError as exc:
        console.print(
            Panel(
                "\n".join(validation_details(exc)),
                title="Configuration not saved",
                border_style="red",
            )
        )
        return
    console.print("Configuration saved.")


def _import(console: Console, session: WorkspaceSession) -> None:
    """Import provider settings from a JSON file."""
    raw_path = Prompt.ask(
        "Provider settings JSON path",
        default=str(session.paths.exports.parent / "provider-settings.json"),
        console=console,
    )
    path = Path(raw_path).expanduser()
    try:
        raw = path.read_text(encoding="utf-8")
        config = apply_provider_settings(session.config, raw)
        session.save_config(config)
    except OSError as exc:
        console.print(
            f"Cannot read {sanitize_terminal_text(str(path))}: {exc.strerror}"
        )
        return
    except (AtlasReviewError, ValidationError) as exc:
        details = (
            exc.details
            if isinstance(exc, AtlasReviewError)
            else validation_details(exc)
        )
        console.print(
            Panel(
                "\n".join(details),
                title="Provider settings not imported",
                border_style="red",
            )
        )
        return
    console.print("Provider settings imported.")


def _import_app(console: Console, session: WorkspaceSession) -> None:
    """Import one complete application configuration document."""
    raw_path = Prompt.ask(
        "Application configuration JSON path",
        default=str(session.paths.exports / "configuration.json"),
        console=console,
    )
    path = Path(raw_path).expanduser()
    try:
        config = config_from_json(path.read_text(encoding="utf-8"))
        session.save_config(config)
    except OSError as exc:
        console.print(
            f"Cannot read {sanitize_terminal_text(str(path))}: {exc.strerror}"
        )
        return
    except ValidationError as exc:
        console.print(
            Panel(
                "\n".join(validation_details(exc)),
                title="Configuration not imported",
                border_style="red",
            )
        )
        return
    console.print("Configuration imported.")


def _write(console: Console, target: Path, text: str) -> None:
    """Write one configuration document into the workspace exports."""
    try:
        target.write_text(text, encoding="utf-8")
    except OSError:
        print_error(
            console,
            AtlasReviewError(
                code="export_failed",
                stage="export",
                message=f"cannot write {sanitize_terminal_text(str(target))}",
                recovery="Choose a writable workspace and try again.",
            ),
        )
        return
    console.print(f"Wrote {sanitize_terminal_text(str(target))}")


def _provider_json(config: LLMConfig) -> str:
    """Return the documented provider-settings document without the secret."""
    payload = {
        "provider_name": config.provider_name,
        "base_url": config.base_url,
        "model": config.model,
        "api_key_mode": config.api_key_mode,
        "api_key_env": config.api_key_env,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _credential_present(config: LLMConfig) -> bool:
    """Return whether the configured variable is currently set."""
    return bool(config.api_key_env and os.environ.get(config.api_key_env, "").strip())
