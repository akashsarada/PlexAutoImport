"""Reliable file movement shared by importing and event grouping."""

import errno
import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)


def move_file(src: str, dest: str) -> bool:
    """Move *src* to *dest* when valid; return whether the move completed."""
    if not os.path.exists(src):
        logger.error("Source file does not exist: %s", src)
        return False
    if os.path.exists(dest):
        logger.warning("Destination file already exists, skipping: %s", dest)
        return False

    file_size = os.path.getsize(src)
    started_at = time.perf_counter()
    mode = robust_move(src, dest)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "Moved %s -> %s mode=%s bytes=%d elapsed_ms=%.1f",
        src,
        dest,
        mode,
        file_size,
        elapsed_ms,
    )
    return True


def robust_move(src: str, dest: str) -> str:
    """Rename on one filesystem, otherwise copy then delete the source."""
    try:
        os.rename(src, dest)
        return "rename"
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    shutil.copy2(src, dest)
    try:
        os.remove(src)
    except PermissionError:
        time.sleep(1)
        os.remove(src)
    return "copy"
