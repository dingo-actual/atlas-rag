"""Deterministic hashing helpers for content fingerprints and record identifiers."""

from __future__ import annotations

import hashlib
import json

from atlas_review.contracts import JsonValue


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of raw bytes.

    Args:
        data (bytes): Bytes to digest.

    Returns:
        (str) Hex digest without a prefix.
    """
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: JsonValue) -> str:
    """Serialize a JSON value so equal values always produce equal text.

    Args:
        value (JsonValue): Plain JSON-compatible value.

    Returns:
        (str) Canonical JSON with sorted keys and no insignificant spacing.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(value: JsonValue) -> str:
    """Return the SHA-256 digest of a canonical JSON value.

    Args:
        value (JsonValue): Plain JSON-compatible value.

    Returns:
        (str) Hex digest of the canonical encoding.
    """
    return sha256_hex(canonical_json(value).encode("utf-8"))
