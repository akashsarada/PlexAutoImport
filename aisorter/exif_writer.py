"""Read and write EXIF/XMP keyword tags on images and videos via PyExifTool."""

import os
import sys

# Ensure parent directory is in sys.path when run as a script directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exiftool import ExifToolHelper

from constants import VIDEO_EXTENSIONS


def write_keywords(image_path: str, keywords: list[str]) -> None:
    """Write keyword tags to an image or video file."""
    if not keywords:
        return

    ext = os.path.splitext(image_path)[1].lower()

    if ext in VIDEO_EXTENSIONS:
        tags = {
            "XMP:Subject": keywords,
            "Keys:Keywords": keywords,
            "ItemList:Keyword": keywords,
        }
    else:
        tags = {
            "EXIF:Keywords": keywords,
            "XMP:Subject": keywords,
        }

    try:
        with ExifToolHelper() as et:
            # -overwrite_original avoids backup copies on the NAS; -P preserves the modification date
            et.set_tags(image_path, tags=tags, params=["-overwrite_original", "-P"])
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Failed to write keywords %s to %s: %s", keywords, image_path, e)


def read_keywords(image_path: str) -> list[str]:
    """Read keyword tags from any of the standard EXIF/XMP locations."""
    with ExifToolHelper() as et:
        tags = et.get_tags(image_path, ["EXIF:Keywords", "XMP:Subject", "Keys:Keywords", "ItemList:Keyword"])[0]
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
