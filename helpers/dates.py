"""Derive a media file's capture date consistently across the importer and event sorter."""

import datetime
import logging
import os
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ExifDateReader = Callable[[str], Optional[datetime.date]]
_EXIF_DATE_PATTERN = re.compile(r"^(\d{4})[:-](\d{2})[:-](\d{2})")


def parse_exif_date(value: object) -> Optional[datetime.date]:
    """Parse an exiftool date string such as '2019:06:14 12:34:56' into a date."""
    if not isinstance(value, str):
        return None
    match = _EXIF_DATE_PATTERN.match(value)
    if match is None:
        return None
    try:
        return datetime.date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def date_from_filename(path: str) -> Optional[datetime.date]:
    """Return the date encoded in a YYYYMMDD_* filename prefix, or None."""
    prefix = os.path.basename(path).split("_")[0]
    if len(prefix) != 8 or not prefix.isdigit():
        return None
    try:
        return datetime.datetime.strptime(prefix, "%Y%m%d").date()
    except ValueError:
        return None


def file_date(path: str, exif_reader: Optional[ExifDateReader] = None) -> datetime.date:
    """Capture date from EXIF when a reader is supplied, else the filename prefix, else mtime."""
    if exif_reader is not None:
        try:
            exif_date = exif_reader(path)
        except Exception:
            logger.exception("EXIF date read failed for %s", path)
            exif_date = None
        if exif_date is not None:
            return exif_date

    filename_date = date_from_filename(path)
    if filename_date is not None:
        return filename_date

    return datetime.date.fromtimestamp(os.path.getmtime(path))
