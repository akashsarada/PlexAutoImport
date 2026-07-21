"""Import photos and videos: tag with AI-detected keywords, then file into year folders."""

import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile
import time

from aisorter.exif_writer import write_keywords
from aisorter.pipeline import AISorterPipeline
from aisorter.preprocess_videos import extract_frames_from_video
from constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


def move_file(src: str, dest: str) -> None:
    if not os.path.exists(src):
        logger.error("Source file does not exist: %s", src)
        return
    if os.path.exists(dest):
        logger.warning("Destination file already exists, skipping: %s", dest)
        return
    logger.info("Moving %s -> %s", src, dest)
    robust_move(src, dest)


def robust_move(src: str, dst: str) -> None:
    """Copy-then-delete move that retries once on Windows file locks."""
    shutil.copy2(src, dst)
    time.sleep(0.1)
    try:
        os.remove(src)
    except PermissionError:
        time.sleep(1)
        os.remove(src)


def _tag_video(pipeline: AISorterPipeline, file_path: str, file_name: str) -> None:
    all_categories: set[str] = set()
    all_identities: set[str] = set()

    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = os.path.splitext(file_name)[0]
        saved_frames = extract_frames_from_video(file_path, temp_dir, base_name)
        for frame_path in saved_frames:
            try:
                res = pipeline.process_image(frame_path)
                all_categories.update(res["categories"])
                all_identities.update(res["identities"])
            except Exception:
                logger.exception("Failed to process frame %s", frame_path)

    labels = list(all_identities) + [c for c in all_categories if c not in all_identities]
    logger.info("Labels for %s: %s", file_name, labels)
    write_keywords(file_path, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag photos/videos with AI keywords and sort them into year folders."
    )
    parser.add_argument("src", help="Source folder containing media to import")
    parser.add_argument("dest", help="Destination root; files land in 'Photos from <year>' subfolders")
    parser.add_argument("model", help="Path to the category classifier ONNX model")
    parser.add_argument("--references", default=None, help="Directory of per-person reference face folders")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if not os.path.exists(args.src):
        logger.error("Source folder does not exist: %s", args.src)
        return 1
    os.makedirs(args.dest, exist_ok=True)

    pipeline = AISorterPipeline(
        reference_dir=args.references,
        category_model_path=args.model,
    )

    error_count = 0
    for file in list(os.listdir(args.src)):
        file_path = os.path.join(args.src, file)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(file)[1].lower()
        creation_year = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).year

        try:
            if ext in IMAGE_EXTENSIONS:
                pipeline.process_image(file_path)
            elif ext in VIDEO_EXTENSIONS:
                _tag_video(pipeline, file_path, file)
        except Exception:
            error_count += 1
            logger.exception("Failed to process %s; moving it untagged", file)

        dest_folder = os.path.join(args.dest, f"Photos from {creation_year}")
        os.makedirs(dest_folder, exist_ok=True)
        move_file(file_path, os.path.join(dest_folder, file))

    if error_count:
        logger.warning("%d file(s) failed AI processing but were still moved", error_count)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
