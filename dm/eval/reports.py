"""Small, strict primitives shared by side-report writers."""

from __future__ import annotations

import math
from pathlib import Path


def json_safe(value):
    """Replace unavailable non-finite readings with JSON ``null`` recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def staged_path(path: Path) -> Path:
    """Same-directory temporary path required for atomic replacement."""
    return path.with_name(f".{path.name}.tmp")
