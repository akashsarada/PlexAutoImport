"""Group date-prefixed photos into per-day event folders when enough photos share a date."""

import argparse
import logging
import os
import sys
from typing import Optional

import main

logger = logging.getLogger(__name__)


def date_prefix(filename: str) -> Optional[str]:
    """Return the leading date token of an underscore-separated filename, or None."""
    prefix = filename.split("_")[0]
    return prefix if len(prefix) == 8 and prefix.isdigit() else None


def group_events(directory: str, threshold: int) -> None:
    window: list[str] = []
    for file in sorted(os.listdir(directory)):
        if not os.path.isfile(os.path.join(directory, file)):
            continue
        if date_prefix(file) is None:
            logger.warning("Skipping file without date prefix: %s", file)
            continue

        window.append(file)
        if len(window) <= threshold:
            continue

        first_date = date_prefix(window[0])
        last_date = date_prefix(window[-1])
        logger.info("Window of %d files spans %s to %s", threshold, first_date, last_date)

        event_folder = os.path.join(directory, f"Event on {first_date}")
        if first_date == last_date and not os.path.exists(event_folder):
            os.makedirs(event_folder)
        if os.path.exists(event_folder):
            main.move_file(
                os.path.join(directory, window[0]),
                os.path.join(event_folder, window[0]),
            )
        window.pop(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle consecutive same-day photos into 'Event on <date>' folders."
    )
    parser.add_argument("src", help="Folder of date-prefixed photos (e.g. 20190614_123456.jpg)")
    parser.add_argument("threshold", type=int, help="Minimum photos on one day to form an event")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not os.path.isdir(args.src):
        logger.error("Source folder does not exist: %s", args.src)
        sys.exit(1)
    group_events(args.src, args.threshold)
