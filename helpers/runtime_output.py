"""Configure persistent logging and terminal progress output."""

import logging
import sys
from pathlib import Path
from typing import Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")
_HANDLER_MARKER = "plex_auto_import"
_verbose_terminal = False


def configure_logging(verbose: bool, log_file: str) -> Path:
    """Log every INFO+ message to a file and optionally mirror it to the terminal."""
    global _verbose_terminal
    _verbose_terminal = verbose
    resolved_log_file = Path(log_file).expanduser().resolve()
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, "name", None) == _HANDLER_MARKER:
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
    file_handler.name = _HANDLER_MARKER
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.name = _HANDLER_MARKER
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root_logger.addHandler(stream_handler)

    return resolved_log_file


def is_verbose() -> bool:
    """Return whether terminal logs replace progress bars for this run."""
    return _verbose_terminal


def progress(
    items: Iterable[T],
    *,
    verbose: bool,
    description: str,
    unit: str,
) -> Iterable[T]:
    """Show a progress bar in quiet mode; verbose mode uses terminal logs instead."""
    return tqdm(
        items,
        desc=description,
        unit=unit,
        disable=verbose,
        dynamic_ncols=True,
    )
