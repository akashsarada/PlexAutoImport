import os
from exiftool import ExifToolHelper
from aisorter.preprocess_videos import VIDEO_EXTENSIONS

def write_keywords(image_path: str, keywords: list[str]) -> None:
    """Write tags to an image or video file using PyExifTool."""
    ext = os.path.splitext(image_path)[1].lower()

    # Define appropriate tags for the file type
    if ext in VIDEO_EXTENSIONS:
        tags = {
            "XMP:Subject": keywords,
            "Keys:Keywords": keywords,
            "ItemList:Keyword": keywords
        }
    else:
        tags = {
            "EXIF:Keywords": keywords,
            "XMP:Subject": keywords
        }

    with ExifToolHelper() as et:
        et.set_tags(
            image_path,
            tags=tags,
            params=["-overwrite_original", "-P"]  # Prevents creating file backups on your NAS and preserves file modification date
        )

# Maintain the exact alias requested by the user
write_tags_to_file = write_keywords


def read_keywords(image_path: str) -> list[str]:
    """Read EXIF and XMP tags from an image or video file using PyExifTool."""
    with ExifToolHelper() as et:
        tags = et.get_tags(image_path, ["EXIF:Keywords", "XMP:Subject", "Keys:Keywords", "ItemList:Keyword"])[0]
        # Get keywords from any of the standard locations
        keywords = (
            tags.get("EXIF:Keywords") or
            tags.get("XMP:Subject") or
            tags.get("Keys:Keywords") or
            tags.get("ItemList:Keyword") or
            []
        )
        if isinstance(keywords, str):
            return [keywords]
        return list(keywords)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 exif_writer.py <image_or_video_path>")
        sys.exit(1)

    path = sys.argv[1].removeprefix("[").removesuffix("]")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Reading keywords from: {path}")
    keywords = read_keywords(path)
    print(f"Keywords: {keywords}")
