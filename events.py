"""Group date-prefixed media into per-day event folders."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Callable, Optional

from constants import DEFAULT_EVENT_THRESHOLD
from moving import move_file

logger = logging.getLogger(__name__)

MoveFile = Callable[[str, str], bool]


def date_prefix(filename: str) -> Optional[str]:
    """Return the leading date token of an underscore-separated filename, or None."""
    prefix = filename.split("_")[0]
    return prefix if len(prefix) == 8 and prefix.isdigit() else None


def group_events(
    directory: str,
    threshold: int = DEFAULT_EVENT_THRESHOLD,
    move: MoveFile = move_file,
) -> int:
    """Group same-date files when their count meets *threshold*; return files moved."""
    if threshold < 1:
        raise ValueError("Event threshold must be at least 1")

    files_by_date: dict[str, list[str]] = defaultdict(list)
    for file in sorted(os.listdir(directory)):
        if not os.path.isfile(os.path.join(directory, file)):
            continue
        file_date = date_prefix(file)
        if file_date is None:
            logger.warning("Skipping file without date prefix: %s", file)
            continue
        files_by_date[file_date].append(file)

    moved_count = 0
    for file_date, files in files_by_date.items():
        if len(files) < threshold:
            logger.info(
                "Event on %s skipped: %d file(s), threshold=%d",
                file_date,
                len(files),
                threshold,
            )
            continue

        event_folder = os.path.join(directory, f"Event on {file_date}")
        os.makedirs(event_folder, exist_ok=True)
        logger.info("Grouping event on %s: %d file(s)", file_date, len(files))
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
    parser.add_argument("src", help="Folder of date-prefixed media (e.g. 20190614_123456.jpg)")
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
    try:
        group_events(args.src, args.threshold)
    except ValueError as error:
        logger.error("%s", error)
        sys.exit(2)
