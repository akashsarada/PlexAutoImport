"""Read and write EXIF/XMP keyword tags via a reusable ExifTool process."""

import logging
import os
import sys
from typing import Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exiftool import ExifToolHelper

from constants import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)
_WRITE_PARAMS = ["-overwrite_original", "-P"]


def _keyword_tags(image_path: str, keywords: list[str]) -> dict[str, list[str]]:
    """Build format-appropriate keyword tags for one media file."""
    if os.path.splitext(image_path)[1].lower() in VIDEO_EXTENSIONS:
        return {
            "XMP:Subject": keywords,
            "Keys:Keywords": keywords,
            "ItemList:Keyword": keywords,
        }
    return {
        "EXIF:Keywords": keywords,
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
        try:
            helper = self._get_helper()
            helper.set_tags(
                image_path,
                tags=_keyword_tags(image_path, keywords),
                params=_WRITE_PARAMS,
            )
        except Exception as error:
            logger.warning("Failed to write keywords %s to %s: %s", keywords, image_path, error)
            self._close(None, None, None)
            raise

    def _get_helper(self) -> ExifToolHelper:
        if self._helper is None:
            context = ExifToolHelper()
            self._context = context
            self._helper = context.__enter__()
        return self._helper

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
    """Read keyword tags from any of the standard EXIF/XMP locations."""
    with ExifToolHelper() as helper:
        tags = helper.get_tags(
            image_path,
            ["EXIF:Keywords", "XMP:Subject", "Keys:Keywords", "ItemList:Keyword"],
        )[0]
        keywords = (
            tags.get("EXIF:Keywords")
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
