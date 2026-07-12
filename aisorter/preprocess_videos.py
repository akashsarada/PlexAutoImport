import os
import cv2
import shutil

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v')

def extract_frames_from_video(video_path, output_dir, base_name):
    """
    Extracts frames from a video:
    - If video length < 60 seconds: extract frames at 30%, 60%, 90% of duration.
    - If video length >= 60 seconds: extract a frame every 30 seconds.
    """
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

    if duration_seconds < 30.0:
        # For videos less than a minute, get frames at 30%, 60%, and 90%
        percentages = [0.30, 0.60, 0.90]
        for p in percentages:
            idx = int(total_frames * p)
            # Ensure index is within range
            idx = min(max(0, idx), total_frames - 1)
            frame_indices.append(idx)
    else:
        # For videos >= 60 seconds, extract a frame every 30 seconds
        step_frames = int(10 * fps)
        current_frame = step_frames
        while current_frame < total_frames:
            frame_indices.append(current_frame)
            current_frame += step_frames

    saved_files = []
    for count, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_filename = f"{base_name}_frame_{count+1}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, frame)
            saved_files.append(out_path)
    
    cap.release()
    return saved_files

def preprocess_labeled_folder(labeled_dir):
    """
    Scans the labeled directory. For any class subfolder (e.g., Cars, People, etc.),
    it extracts frames from any video file and saves them as images, then deletes/moves the original video.
    """
    if not os.path.exists(labeled_dir):
        print(f"Error: '{labeled_dir}' directory does not exist.")
        return

    print("Scanning labeled directory for video preprocessing...")
    video_count = 0
    extracted_count = 0

    for root, dirs, files in os.walk(labeled_dir):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                
                print(f"Processing video: {video_path}")
                # Check if frames already exist for this video
                first_frame_path = os.path.join(root, f"{base_name}_frame_1.jpg")
                if os.path.exists(first_frame_path):
                    print(f"  Frames already extracted for {file}, skipping.")
                    video_count += 1
                    continue

                saved_frames = extract_frames_from_video(video_path, root, base_name)
                
                if saved_frames:
                    print(f"  Extracted {len(saved_frames)} frames.")
                    extracted_count += len(saved_frames)
                else:
                    print(f"  Failed to extract any frames from {file}")
                
                video_count += 1

    print(f"Preprocessing completed. Processed {video_count} videos, extracted {extracted_count} frames.")

if __name__ == '__main__':
    # This can be run stand-alone to preprocess dataset/labeled folder
    preprocess_labeled_folder('labeled')
