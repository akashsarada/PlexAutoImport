"""Reliable file movement shared by importing and event grouping."""

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
    robust_move(src, dest)
    logger.info("Moved %s -> %s", src, dest)
    return True


def robust_move(src: str, dest: str) -> None:
    """Copy-then-delete move that retries once on Windows file locks."""
    shutil.copy2(src, dest)
    time.sleep(0.1)
    try:
        os.remove(src)
    except PermissionError:
        time.sleep(1)
        os.remove(src)
