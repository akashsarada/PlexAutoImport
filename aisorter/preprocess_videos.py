"""Extract representative frames from videos for classification and training."""

import argparse
import os

import cv2

from constants import DURATION_THRESHOLD_SECONDS, VIDEO_EXTENSIONS


def extract_frames_from_video(video_path: str, output_dir: str, base_name: str) -> list[str]:
    """Extract frames: 3 spread frames for short clips, one every 2 seconds otherwise."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    duration_seconds = total_frames / fps
    frame_indices = []

    if duration_seconds < DURATION_THRESHOLD_SECONDS:
        for p in (0.10, 0.40, 0.70):
            idx = min(max(0, int(total_frames * p)), total_frames - 1)
            frame_indices.append(idx)
    else:
        step_frames = int(2 * fps)
        current_frame = int(0.75 * fps)
        while current_frame < total_frames:
            frame_indices.append(current_frame)
            current_frame += step_frames

    saved_files = []
    for count, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"{base_name}_frame_{count + 1}.jpg")
            cv2.imwrite(out_path, frame)
            saved_files.append(out_path)

    cap.release()
    return saved_files


def preprocess_labeled_folder(labeled_dir: str) -> None:
    """Extract frames from every video under each class subfolder of *labeled_dir*."""
    if not os.path.exists(labeled_dir):
        print(f"Error: '{labeled_dir}' directory does not exist.")
        return

    print("Scanning labeled directory for video preprocessing...")
    video_count = 0
    extracted_count = 0

    for root, _, files in os.walk(labeled_dir):
        for file in files:
            if not file.lower().endswith(VIDEO_EXTENSIONS):
                continue
            video_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]
            video_count += 1

            print(f"Processing video: {video_path}")
            if os.path.exists(os.path.join(root, f"{base_name}_frame_1.jpg")):
                print(f"  Frames already extracted for {file}, skipping.")
                continue

            saved_frames = extract_frames_from_video(video_path, root, base_name)
            if saved_frames:
                print(f"  Extracted {len(saved_frames)} frames.")
                extracted_count += len(saved_frames)
            else:
                print(f"  Failed to extract any frames from {file}")

    print(f"Preprocessing completed. Processed {video_count} videos, extracted {extracted_count} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in a labeled dataset folder.")
    parser.add_argument("labeled_dir", nargs="?", default="labeled", help="Labeled dataset root (default: labeled)")
    args = parser.parse_args()
    preprocess_labeled_folder(args.labeled_dir)
