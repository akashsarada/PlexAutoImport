"""Import photos and videos: tag with AI-detected keywords, then file into year folders."""

import argparse
import datetime
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aisorter.exif_writer import write_keywords
from aisorter.pipeline import AISorterPipeline
from aisorter.preprocess_videos import extract_frames_from_video
from helpers.runtime_output import configure_logging, progress
from constants import DEFAULT_EVENT_THRESHOLD, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from events import group_events
from helpers.moving import move_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    src: str
    dest: str
    category_model: str
    face_detector_model: Optional[str]
    references: Optional[str]
    face_identifier_model: Optional[str]
    event_threshold: int = DEFAULT_EVENT_THRESHOLD
    family_group: str = "Family"
    family_dest: Optional[str] = None


@dataclass(frozen=True)
class ImportStats:
    images_sorted: int
    videos_sorted: int
    elapsed_seconds: float
    event_files_grouped: int = 0
    new_people_found: int = 0
    family_photos_sorted: int = 0

    @property
    def total_entities(self) -> int:
        return self.images_sorted + self.videos_sorted

    @property
    def entities_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_entities / self.elapsed_seconds


def _tag_video(pipeline: AISorterPipeline, file_path: str, file_name: str) -> dict:
    """Tag a video with AI keywords; returns a result dict including is_family_photo."""
    all_categories: set[str] = set()
    all_identities: set[str] = set()
    is_family_photo = False

    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = os.path.splitext(file_name)[0]
        saved_frames = extract_frames_from_video(file_path, temp_dir, base_name)
        for frame_path in saved_frames:
            try:
                res = pipeline.process_image(frame_path)
                all_categories.update(res["categories"])
                all_identities.update(res["identities"])
                if res.get("is_family_photo"):
                    is_family_photo = True
            except Exception:
                logger.exception("Failed to process frame %s", frame_path)

    labels = list(all_identities) + [c for c in all_categories if c not in all_identities]
    logger.info("Labels for %s: %s", file_name, labels)
    write_keywords(file_path, labels)
    return {"identities": list(all_identities), "is_family_photo": is_family_photo}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag photos/videos with AI keywords and sort them into year folders."
    )
    parser.add_argument("src", nargs="?", help="Source folder containing media to import")
    parser.add_argument("dest", nargs="?", help="Destination root for year folders")
    parser.add_argument("model", nargs="?", help="Category classifier ONNX model")
    parser.add_argument("--face-detector-model", help="Face detector ONNX model")
    parser.add_argument("--references", help="Directory of per-person reference face folders")
    parser.add_argument("--face-identifier-model", help="Face identifier ONNX model")
    parser.add_argument(
        "--event-threshold",
        type=int,
        default=DEFAULT_EVENT_THRESHOLD,
        help=f"Minimum same-day files needed for an event (default: {DEFAULT_EVENT_THRESHOLD})",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt for missing locations (default: enabled)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print logs instead of the progress bar; logs are always written to file",
    )
    parser.add_argument(
        "--log-file",
        default="plex_auto_import.log",
        help="Log file path (default: plex_auto_import.log)",
    )
    parser.add_argument(
        "--family-group",
        default="Family",
        help="Name of the references group subfolder treated as family (default: Family)",
    )
    parser.add_argument(
        "--family-dest",
        default=None,
        help="Destination folder for photos with 2+ family members (default: <dest>/Family Photos)",
    )
    return parser.parse_args(argv)


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
        f"Event files grouped: {stats.event_files_grouped}",
        f"Family photos sorted: {stats.family_photos_sorted}",
        f"New people found: {stats.new_people_found}",
        f"Elapsed time: {stats.elapsed_seconds:.2f} seconds",
        f"Entities per second: {stats.entities_per_second:.2f}",
    )
    logger.info("Import statistics")
    for line in lines:
        logger.info(line)
    if not verbose:
        print("\nImport statistics")
        for line in lines:
            print(line)


def run_import(config: RuntimeConfig, verbose: bool) -> int:
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
    family_photos_sorted = 0
    destination_folders: set[str] = set()
    started_at = time.perf_counter()
    for file in progress(files, verbose=verbose, description="Importing media", unit="file"):
        file_path = os.path.join(config.src, file)
        ext = os.path.splitext(file)[1].lower()
        creation_year = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).year
        is_family = False

        try:
            if ext in IMAGE_EXTENSIONS:
                result = pipeline.process_image(file_path)
                is_family = result.get("is_family_photo", False)
            elif ext in VIDEO_EXTENSIONS:
                result = _tag_video(pipeline, file_path, file)
                is_family = result.get("is_family_photo", False)
        except Exception:
            error_count += 1
            logger.exception("Failed to process %s; moving it untagged", file)

        if is_family:
            dest_folder = family_dest
        else:
            dest_folder = os.path.join(config.dest, f"Photos from {creation_year}")
        os.makedirs(dest_folder, exist_ok=True)
        moved = move_file(file_path, os.path.join(dest_folder, file))
        if moved and ext in IMAGE_EXTENSIONS:
            images_sorted += 1
            destination_folders.add(dest_folder)
            if is_family:
                family_photos_sorted += 1
        elif moved and ext in VIDEO_EXTENSIONS:
            videos_sorted += 1
            destination_folders.add(dest_folder)
            if is_family:
                family_photos_sorted += 1

    event_files_grouped = 0
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
        )

    stats = ImportStats(
        images_sorted=images_sorted,
        videos_sorted=videos_sorted,
        elapsed_seconds=time.perf_counter() - started_at,
        event_files_grouped=event_files_grouped,
        new_people_found=pipeline.new_people_found,
        family_photos_sorted=family_photos_sorted,
    )
    report_stats(stats, verbose)

    if error_count:
        logger.warning("%d file(s) failed AI processing but were still moved", error_count)
        return 1
    logger.info("Import completed successfully: %d media file(s)", stats.total_entities)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose, args.log_file)
    try:
        config = resolve_config(args)
    except ValueError as error:
        logger.error("%s", error)
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return run_import(config, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
