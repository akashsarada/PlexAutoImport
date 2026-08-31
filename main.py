"""Import photos and videos: tag with AI-detected keywords, then file into year folders."""

import argparse
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aisorter.exif_writer import ExifToolKeywordWriter, WriteKeywords, write_keywords
from aisorter.pipeline import AISorterPipeline, merge_labels
from aisorter.preprocess_videos import extract_frames_from_video
from helpers.runtime_output import configure_logging, progress
from constants import (
    DEFAULT_EVENT_THRESHOLD,
    DEFAULT_FAMILY_GROUP,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from events import group_events
from helpers.config_file import load_config_file
from helpers.dates import file_date
from helpers.moving import move_file

logger = logging.getLogger(__name__)
ARGUMENT_DEFAULTS = {
    "event": True,
    "event_threshold": DEFAULT_EVENT_THRESHOLD,
    "interactive": True,
    "verbose": False,
    "log_file": "plex_auto_import.log",
    "family_group": DEFAULT_FAMILY_GROUP,
}


@dataclass(frozen=True)
class RuntimeConfig:
    src: str
    dest: str
    category_model: str
    face_detector_model: Optional[str]
    references: Optional[str]
    face_identifier_model: Optional[str]
    event_threshold: int = DEFAULT_EVENT_THRESHOLD
    family_group: str = DEFAULT_FAMILY_GROUP
    family_dest: Optional[str] = None


@dataclass(frozen=True)
class ImportStats:
    images_sorted: int
    videos_sorted: int
    elapsed_seconds: float
    frames_processed: int = 0
    event_files_grouped: int = 0
    new_people_found: int = 0
    family_photos_sorted: int = 0
    files_skipped: int = 0
    non_media_skipped: int = 0

    @property
    def total_entities(self) -> int:
        return self.images_sorted + self.videos_sorted

    @property
    def entities_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_entities / self.elapsed_seconds

    @property
    def images_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


def _tag_video(
    pipeline: AISorterPipeline,
    file_path: str,
    file_name: str,
    keyword_writer: Optional[WriteKeywords] = None,
) -> dict:
    """Tag a video and return its identities, family status, and processed frame count."""
    all_categories: set[str] = set()
    all_identities: set[str] = set()
    is_family_photo = False
    frames_processed = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = os.path.splitext(file_name)[0]
        saved_frames = extract_frames_from_video(file_path, temp_dir, base_name)
        for frame_path in saved_frames:
            try:
                res = pipeline.process_image(frame_path, write_metadata=False)
                frames_processed += 1
                all_categories.update(res["categories"])
                all_identities.update(res["identities"])
                if res.get("is_family_photo"):
                    is_family_photo = True
            except Exception:
                logger.exception("Failed to process frame %s", frame_path)

    labels = merge_labels(sorted(all_identities), sorted(all_categories))
    logger.info("Labels for %s: %s", file_name, labels)
    (keyword_writer or write_keywords)(file_path, labels)
    return {
        "identities": list(all_identities),
        "is_family_photo": is_family_photo,
        "frames_processed": frames_processed,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag photos/videos with AI keywords and sort them into year folders."
    )
    parser.add_argument("src", nargs="?", help="Source folder containing media to import")
    parser.add_argument("dest", nargs="?", help="Destination root for year folders")
    parser.add_argument("model", nargs="?", help="Category classifier ONNX model")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="JSON config file supplying any input not given on the command line",
    )
    parser.add_argument("--face-detector-model", help="Face detector ONNX model")
    parser.add_argument("--references", help="Directory of per-person reference face folders")
    parser.add_argument("--face-identifier-model", help="Face identifier ONNX model")
    parser.add_argument(
        "--event",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Group media into event folders after import (default: enabled)",
    )
    parser.add_argument(
        "--event-threshold",
        type=int,
        default=None,
        help=f"Minimum same-day files needed for an event (default: {DEFAULT_EVENT_THRESHOLD})",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prompt for missing locations (default: enabled)",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print logs instead of the progress bar; logs are always written to file",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: plex_auto_import.log)",
    )
    parser.add_argument(
        "--family-group",
        default=None,
        help="Name of the references group subfolder treated as family (default: Family)",
    )
    parser.add_argument(
        "--family-dest",
        default=None,
        help="Destination folder for photos with 2+ family members (default: <dest>/Family Photos)",
    )
    args = parser.parse_args(argv)
    return _apply_input_precedence(args)


def _apply_input_precedence(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset arguments from the config file, then from the built-in defaults."""
    if args.config is not None:
        for key, value in load_config_file(args.config).items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    for key, value in ARGUMENT_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def _resolve_path(
    value: Optional[str],
    *,
    label: str,
    interactive: bool,
    prompt: Callable[[str], str],
    required: bool,
    path_type: str,
) -> Optional[str]:
    candidate = value
    while True:
        if candidate is None and interactive:
            suffix = "" if required else " (leave blank for default/disabled)"
            candidate = prompt(f"{label}{suffix}: ").strip() or None

        if candidate is None:
            if required:
                raise ValueError(f"Missing required location: {label}")
            return None

        resolved = str(Path(candidate).expanduser().resolve())
        exists = os.path.isdir(resolved) if path_type == "directory" else os.path.isfile(resolved)
        if path_type == "output" or exists:
            return resolved
        if not interactive or value is not None:
            raise ValueError(f"{label} does not exist: {resolved}")

        print(f"{label} does not exist: {resolved}", file=sys.stderr)
        candidate = None


def resolve_config(
    args: argparse.Namespace,
    prompt: Callable[[str], str] = input,
) -> RuntimeConfig:
    src = _resolve_path(
        args.src,
        label="Source folder",
        interactive=args.interactive,
        prompt=prompt,
        required=True,
        path_type="directory",
    )
    dest = _resolve_path(
        args.dest,
        label="Destination root",
        interactive=args.interactive,
        prompt=prompt,
        required=True,
        path_type="output",
    )
    category_model = _resolve_path(
        args.model,
        label="Category classifier model",
        interactive=args.interactive,
        prompt=prompt,
        required=True,
        path_type="file",
    )
    face_detector_model = _resolve_path(
        args.face_detector_model,
        label="Face detector model",
        interactive=args.interactive,
        prompt=prompt,
        required=False,
        path_type="file",
    )
    references = _resolve_path(
        args.references,
        label="Reference faces directory",
        interactive=args.interactive,
        prompt=prompt,
        required=False,
        path_type="directory",
    )
    face_identifier_model = None
    if references is not None:
        face_identifier_model = _resolve_path(
            args.face_identifier_model,
            label="Face identifier model",
            interactive=args.interactive,
            prompt=prompt,
            required=False,
            path_type="file",
        )

    if args.event_threshold < 1:
        raise ValueError("Event threshold must be at least 1")

    return RuntimeConfig(
        src=src,
        dest=dest,
        category_model=category_model,
        face_detector_model=face_detector_model,
        references=references,
        face_identifier_model=face_identifier_model,
        event_threshold=args.event_threshold,
        family_group=args.family_group,
        family_dest=args.family_dest,
    )


def report_stats(stats: ImportStats, verbose: bool) -> None:
    lines = (
        f"Images sorted: {stats.images_sorted}",
        f"Videos sorted: {stats.videos_sorted}",
        f"Total entities: {stats.total_entities}",
        f"Frames processed: {stats.frames_processed}",
        f"Event files grouped: {stats.event_files_grouped}",
        f"Family photos sorted: {stats.family_photos_sorted}",
        f"New people found: {stats.new_people_found}",
        f"Files skipped (already in destination): {stats.files_skipped}",
        f"Non-media files skipped: {stats.non_media_skipped}",
        f"Elapsed time: {stats.elapsed_seconds:.2f} seconds",
        f"Entities per second: {stats.entities_per_second:.2f}",
        f"Images per second: {stats.images_per_second:.2f}",
    )
    logger.info("Import statistics")
    for line in lines:
        logger.info(line)
    if not verbose:
        print("\nImport statistics")
        for line in lines:
            print(line)


def run_import(config: RuntimeConfig, verbose: bool, event: bool = True) -> int:
    os.makedirs(config.dest, exist_ok=True)
    logger.info("Importing media from %s to %s", config.src, config.dest)

    pipeline = AISorterPipeline(
        reference_dir=config.references,
        category_model_path=config.category_model,
        face_detector_model_path=config.face_detector_model,
        face_identifier_model_path=config.face_identifier_model,
        family_group=config.family_group,
    )
    family_dest = config.family_dest or os.path.join(config.dest, "Family Photos")

    files = [
        file
        for file in sorted(os.listdir(config.src))
        if os.path.isfile(os.path.join(config.src, file))
    ]
    error_count = 0
    images_sorted = 0
    videos_sorted = 0
    frames_processed = 0
    family_photos_sorted = 0
    files_skipped = 0
    non_media_skipped = 0
    destination_folders: set[str] = set()
    started_at = time.perf_counter()
    with ExifToolKeywordWriter() as keyword_writer:
        for file in progress(files, verbose=verbose, description="Importing media", unit="file"):
            file_path = os.path.join(config.src, file)
            ext = os.path.splitext(file)[1].lower()
            is_image = ext in IMAGE_EXTENSIONS
            is_video = ext in VIDEO_EXTENSIONS
            if not is_image and not is_video:
                logger.info("Skipping non-media file: %s", file)
                non_media_skipped += 1
                continue
            is_family = False

            try:
                if is_image:
                    result = pipeline.process_image(
                        file_path,
                        keyword_writer=keyword_writer,
                    )
                    frames_processed += 1
                    is_family = result.get("is_family_photo", False)
                else:
                    result = _tag_video(
                        pipeline,
                        file_path,
                        file,
                        keyword_writer=keyword_writer,
                    )
                    frames_processed += result.get("frames_processed", 0)
                    is_family = result.get("is_family_photo", False)
            except Exception:
                error_count += 1
                logger.exception("Failed to process %s; moving it untagged", file)

            if is_family:
                dest_folder = family_dest
            else:
                creation_year = file_date(file_path, keyword_writer.read_capture_date).year
                dest_folder = os.path.join(config.dest, f"Photos from {creation_year}")
            os.makedirs(dest_folder, exist_ok=True)
            try:
                moved = move_file(file_path, os.path.join(dest_folder, file))
            except OSError:
                error_count += 1
                logger.exception("Failed to move %s; leaving it in source", file)
                continue
            if not moved:
                files_skipped += 1
                continue
            destination_folders.add(dest_folder)
            if is_image:
                images_sorted += 1
            else:
                videos_sorted += 1
            if is_family:
                family_photos_sorted += 1

        event_files_grouped = 0
        if event:
            for destination_folder in sorted(destination_folders):
                logger.info(
                    "Running event sorter for %s with threshold %d",
                    destination_folder,
                    config.event_threshold,
                )
                event_files_grouped += group_events(
                    destination_folder,
                    config.event_threshold,
                    move_file,
                    exif_reader=keyword_writer.read_capture_date,
                )
        else:
            logger.info("Event sorter skipped.")

    stats = ImportStats(
        images_sorted=images_sorted,
        videos_sorted=videos_sorted,
        elapsed_seconds=time.perf_counter() - started_at,
        frames_processed=frames_processed,
        event_files_grouped=event_files_grouped,
        new_people_found=pipeline.new_people_found,
        family_photos_sorted=family_photos_sorted,
        files_skipped=files_skipped,
        non_media_skipped=non_media_skipped,
    )
    report_stats(stats, verbose)

    if error_count or files_skipped:
        if error_count:
            logger.warning("%d file(s) failed processing or moving", error_count)
        if files_skipped:
            logger.warning(
                "%d file(s) skipped and left in the source folder", files_skipped
            )
        return 1
    logger.info("Import completed successfully: %d media file(s)", stats.total_entities)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    configure_logging(args.verbose, args.log_file)
    try:
        config = resolve_config(args)
    except ValueError as error:
        logger.error("%s", error)
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return run_import(config, args.verbose, args.event)


if __name__ == "__main__":
    sys.exit(main())
