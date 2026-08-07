"""Read and write EXIF/XMP keyword tags via a reusable ExifTool process."""

import datetime
import logging
import os
import sys
from typing import Callable, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exiftool import ExifToolHelper

from constants import VIDEO_EXTENSIONS
from helpers.dates import parse_exif_date

logger = logging.getLogger(__name__)

WriteKeywords = Callable[[str, list[str]], None]
_WRITE_PARAMS = ["-overwrite_original", "-P", "-m"]
_DATE_TAGS = ("EXIF:DateTimeOriginal", "QuickTime:CreateDate", "XMP:CreateDate")


def _keyword_tags(image_path: str, keywords: list[str]) -> dict[str, list[str]]:
    """Build format-appropriate keyword tags for one media file."""
    if os.path.splitext(image_path)[1].lower() in VIDEO_EXTENSIONS:
        return {
            "XMP:Subject": keywords,
            "Keys:Keywords": keywords,
            "ItemList:Keyword": keywords,
        }
    return {
        "IPTC:Keywords": keywords,
        "XMP:Subject": keywords,
    }


class ExifToolKeywordWriter:
    """Write keywords while reusing one lazily started ExifTool process."""

    def __init__(self) -> None:
        self._context: Optional[ExifToolHelper] = None
        self._helper: Optional[ExifToolHelper] = None

    def __enter__(self) -> "ExifToolKeywordWriter":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self._close(exc_type, exc_value, traceback)

    def __call__(self, image_path: str, keywords: list[str]) -> None:
        self.write(image_path, keywords)

    def write(self, image_path: str, keywords: list[str]) -> None:
        """Write keyword tags, restarting ExifTool after a helper failure."""
        if not keywords:
            return
        tags = _keyword_tags(image_path, keywords)
        try:
            self._get_helper().set_tags(
                image_path,
                tags=tags,
                params=_WRITE_PARAMS,
            )
            return
        except Exception as error:
            self._recover_from_failure(image_path)
            is_video = os.path.splitext(image_path)[1].lower() in VIDEO_EXTENSIONS
            if not (is_video and len(tags) > 1):
                logger.warning(
                    "Failed to write keywords %s to %s: %s", keywords, image_path, error
                )
                raise
            logger.warning(
                "Failed to write full tags to video %s: %s. Retrying with XMP:Subject only.",
                image_path,
                error,
            )

        try:
            self._get_helper().set_tags(
                image_path,
                tags={"XMP:Subject": keywords},
                params=_WRITE_PARAMS,
            )
        except Exception as fallback_error:
            logger.warning("Fallback write to %s also failed: %s", image_path, fallback_error)
            self._recover_from_failure(image_path)
            raise

    def read_capture_date(self, image_path: str) -> Optional[datetime.date]:
        """Read the capture date from standard EXIF/XMP/QuickTime tags, or None."""
        try:
            tags = self._get_helper().get_tags(image_path, list(_DATE_TAGS))[0]
        except Exception as error:
            logger.warning("Failed to read capture date from %s: %s", image_path, error)
            self._close(None, None, None)
            return None
        for tag in _DATE_TAGS:
            parsed = parse_exif_date(tags.get(tag))
            if parsed is not None:
                return parsed
        return None

    def _get_helper(self) -> ExifToolHelper:
        if self._helper is None:
            context = ExifToolHelper()
            self._context = context
            self._helper = context.__enter__()
        return self._helper

    def _recover_from_failure(self, image_path: str) -> None:
        """Restart ExifTool and remove any temporary file it left beside the media file."""
        self._close(None, None, None)
        tmp_path = f"{image_path}_exiftool_tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as clean_error:
                logger.warning(
                    "Failed to clean up temporary file %s: %s", tmp_path, clean_error
                )

    def _close(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        context = self._context
        self._context = None
        self._helper = None
        if context is None:
            return
        try:
            context.__exit__(exc_type, exc_value, traceback)
        except Exception as error:
            logger.warning("Failed to stop ExifTool: %s", error)


def write_keywords(image_path: str, keywords: list[str]) -> None:
    """Write keyword tags using a temporary ExifTool process."""
    with ExifToolKeywordWriter() as writer:
        writer(image_path, keywords)


def read_keywords(image_path: str) -> list[str]:
    """Read keyword tags from any of the standard IPTC/XMP locations."""
    with ExifToolHelper() as helper:
        tags = helper.get_tags(
            image_path,
            ["IPTC:Keywords", "XMP:Subject", "Keys:Keywords", "ItemList:Keyword"],
        )[0]
        keywords = (
            tags.get("IPTC:Keywords")
            or tags.get("XMP:Subject")
            or tags.get("Keys:Keywords")
            or tags.get("ItemList:Keyword")
            or []
        )
        if isinstance(keywords, str):
            return [keywords]
        return list(keywords)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print keyword tags for a media file.")
    parser.add_argument("path", help="Image or video file to read")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"File not found: {args.path}")
    print(f"Keywords: {read_keywords(args.path)}")
