"""Extract representative frames from videos for classification and training."""

import argparse
import logging
import os

import cv2

from constants import DURATION_THRESHOLD_SECONDS, VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path: str, output_dir: str, base_name: str) -> list[str]:
    """Extract frames: 3 spread frames for short clips, one every 2 seconds otherwise."""
    logger.info("Starting frame extraction for %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        logger.warning("Video has invalid frame metadata: %s", video_path)
        return []

    duration_seconds = total_frames / fps
    frame_indices = []

    if duration_seconds < DURATION_THRESHOLD_SECONDS:
        for percentage in (0.10, 0.40, 0.70):
            index = min(max(0, int(total_frames * percentage)), total_frames - 1)
            frame_indices.append(index)
    else:
        step_frames = int((DURATION_THRESHOLD_SECONDS / 3) * fps)
        current_frame = int(0.75 * fps)
        while current_frame < total_frames:
            frame_indices.append(current_frame)
            current_frame += step_frames

    saved_files = []
    for count, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        read_succeeded, frame = cap.read()
        if read_succeeded:
            out_path = os.path.join(output_dir, f"{base_name}_frame_{count + 1}.jpg")
            cv2.imwrite(out_path, frame)
            saved_files.append(out_path)

    cap.release()
    logger.info("Finished frame extraction for %s: extracted %d frames", video_path, len(saved_files))
    return saved_files


def preprocess_labeled_folder(labeled_dir: str) -> None:
    """Extract frames from every video under each class subfolder of *labeled_dir*."""
    if not os.path.exists(labeled_dir):
        logger.error("Labeled directory does not exist: %s", labeled_dir)
        return

    logger.info("Scanning labeled directory for video preprocessing")
    video_count = 0
    extracted_count = 0

    for root, _, files in os.walk(labeled_dir):
        for file in files:
            if not file.lower().endswith(VIDEO_EXTENSIONS):
                continue
            video_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]
            video_count += 1

            logger.info("Processing video: %s", video_path)
            if os.path.exists(os.path.join(root, f"{base_name}_frame_1.jpg")):
                logger.info("Frames already extracted for %s; skipping", file)
                continue

            saved_frames = extract_frames_from_video(video_path, root, base_name)
            if saved_frames:
                logger.info("Extracted %d frame(s)", len(saved_frames))
                extracted_count += len(saved_frames)
            else:
                logger.warning("Failed to extract frames from %s", file)

    logger.info(
        "Preprocessing completed: %d video(s), %d frame(s)",
        video_count,
        extracted_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Extract frames from videos in a labeled dataset folder.")
    parser.add_argument("labeled_dir", nargs="?", default="labeled", help="Labeled dataset root (default: labeled)")
    args = parser.parse_args()
    preprocess_labeled_folder(args.labeled_dir)
