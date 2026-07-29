"""Load importer inputs from a JSON configuration file."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PATH_KEYS = frozenset(
    {
        "src",
        "dest",
        "model",
        "face_detector_model",
        "references",
        "face_identifier_model",
        "family_dest",
        "log_file",
    }
)
STRING_KEYS = frozenset({"family_group"})
BOOLEAN_KEYS = frozenset({"event", "interactive", "verbose"})
INTEGER_KEYS = frozenset({"event_threshold"})
CONFIG_KEYS = PATH_KEYS | STRING_KEYS | BOOLEAN_KEYS | INTEGER_KEYS


def load_config_file(config_path: str) -> dict[str, Any]:
    """Return the inputs declared in the JSON config file at *config_path*."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Config file does not exist: {path}")

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON: {path}: {error}") from error
    except OSError as error:
        raise ValueError(f"Config file could not be read: {path}: {error}") from error

    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    unknown_keys = sorted(set(content) - CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) in {path}: {', '.join(unknown_keys)}. "
            f"Supported keys: {', '.join(sorted(CONFIG_KEYS))}"
        )

    values = {
        key: _coerce_value(key, value, path)
        for key, value in content.items()
        if value is not None
    }
    logger.info("Loaded config file %s: keys=%s", path, sorted(values))
    return values


def _coerce_value(key: str, value: Any, config_path: Path) -> Any:
    if key in BOOLEAN_KEYS:
        if not isinstance(value, bool):
            raise ValueError(f"Config key '{key}' must be true or false in {config_path}")
        return value

    if key in INTEGER_KEYS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Config key '{key}' must be an integer in {config_path}")
        return value

    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string in {config_path}")

    if key in PATH_KEYS:
        return _absolute_path(value, config_path.parent)
    return value


def _absolute_path(value: str, base_dir: Path) -> str:
    """Resolve *value* against *base_dir* so config files stay portable."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return os.path.normpath(str(candidate))
