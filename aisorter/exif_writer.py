import piexif

_XP_KEYWORDS_TAG = 0x9C9E


def _is_jpeg(image_path: str) -> bool:
    with open(image_path, "rb") as f:
        return f.read(2) == b"\xff\xd8"


def _load_exif(image_path: str) -> dict:
    try:
        return piexif.load(image_path)
    except Exception:
        return {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}


def write_keywords(image_path: str, keywords: list[str]) -> None:
    if not _is_jpeg(image_path):
        raise ValueError(f"Not a JPEG file: {image_path}")

    exif_dict = _load_exif(image_path)
    encoded = ";".join(keywords).encode("utf-16-le")

    ifd = exif_dict.setdefault("0th", {})
    tag = piexif.ImageIFD.XPKeywords if hasattr(piexif.ImageIFD, "XPKeywords") else _XP_KEYWORDS_TAG
    ifd[tag] = encoded

    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)


def read_keywords(image_path: str) -> list[str]:
    if not _is_jpeg(image_path):
        raise ValueError(f"Not a JPEG file: {image_path}")

    try:
        exif_dict = piexif.load(image_path)
    except Exception:
        return []

    ifd = exif_dict.get("0th", {})
    tag = piexif.ImageIFD.XPKeywords if hasattr(piexif.ImageIFD, "XPKeywords") else _XP_KEYWORDS_TAG
    raw = ifd.get(tag)

    if not raw:
        return []

    decoded = raw.decode("utf-16-le").rstrip("\x00")
    return [k for k in decoded.split(";") if k]
