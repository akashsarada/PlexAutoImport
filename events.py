"""Group media into per-day event folders using the shared capture-date derivation."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Callable, Optional

from constants import DEFAULT_EVENT_THRESHOLD
from helpers.dates import ExifDateReader, file_date
from helpers.moving import move_file

logger = logging.getLogger(__name__)

MoveFile = Callable[[str, str], bool]


def group_events(
    directory: str,
    threshold: int = DEFAULT_EVENT_THRESHOLD,
    move: MoveFile = move_file,
    exif_reader: Optional[ExifDateReader] = None,
) -> int:
    """Group same-date files when their count meets *threshold*; return files moved."""
    if threshold < 1:
        raise ValueError("Event threshold must be at least 1")

    files_by_date: dict[str, list[str]] = defaultdict(list)
    for file in sorted(os.listdir(directory)):
        path = os.path.join(directory, file)
        if not os.path.isfile(path):
            continue
        file_key = file_date(path, exif_reader).strftime("%Y%m%d")
        files_by_date[file_key].append(file)

    moved_count = 0
    for file_key, files in files_by_date.items():
        if len(files) < threshold:
            logger.info(
                "Event on %s skipped: %d file(s), threshold=%d",
                file_key,
                len(files),
                threshold,
            )
            continue

        event_folder = os.path.join(directory, f"Event on {file_key}")
        os.makedirs(event_folder, exist_ok=True)
        logger.info("Grouping event on %s: %d file(s)", file_key, len(files))
        for file in files:
            if move(
                os.path.join(directory, file),
                os.path.join(event_folder, file),
            ):
                moved_count += 1

    logger.info("Event sorting complete for %s: %d file(s) grouped", directory, moved_count)
    return moved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle same-day media into 'Event on <date>' folders."
    )
    parser.add_argument("src", help="Folder of media files to group into per-day events")
    parser.add_argument(
        "threshold",
        nargs="?",
        type=int,
        default=DEFAULT_EVENT_THRESHOLD,
        help=f"Minimum same-day files needed (default: {DEFAULT_EVENT_THRESHOLD})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not os.path.isdir(args.src):
        logger.error("Source folder does not exist: %s", args.src)
        sys.exit(1)
    from aisorter.exif_writer import ExifToolKeywordWriter

    try:
        with ExifToolKeywordWriter() as writer:
            group_events(args.src, args.threshold, exif_reader=writer.read_capture_date)
    except ValueError as error:
        logger.error("%s", error)
        sys.exit(2)
