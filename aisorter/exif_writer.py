from exiftool import ExifToolHelper

def write_keywords(image_path: str, keywords: list[str]) -> None:
    """Write tags to an image file using PyExifTool."""
    with ExifToolHelper() as et:
        et.set_tags(
            image_path,
            tags={
                "EXIF:Keywords": keywords,
                "XMP:Subject": keywords      # PhotoPrism reads both EXIF and XMP tags
            },
            params=["-overwrite_original"]  # Prevents creating file backups on your NAS
        )

def read_keywords(image_path: str) -> list[str]:
    """Read EXIF and XMP tags from an image file using PyExifTool."""
    with ExifToolHelper() as et:
        tags = et.get_tags(image_path, ["EXIF:Keywords", "XMP:Subject"])[0]
        # et.get_tags returns a list of dicts, one per file
        keywords = tags.get("EXIF:Keywords") or tags.get("XMP:Subject") or []
        if isinstance(keywords, str):
            return [keywords]
        return list(keywords)
