import datetime
import os
import shutil
import sys
import tempfile
import time

from aisorter.exif_writer import write_keywords
from aisorter.pipeline import AISorterPipeline
from aisorter.preprocess_videos import extract_frames_from_video, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def move_file(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            print(f"Destination file already exists: {dest}")
        else:
            print(f"Moving file {src} to {dest}")
            robust_move(src, dest)
    else:
        print(f"Source file does not exist: {src}")


def robust_move(src, dst):
    # 1. Copy the file first
    shutil.copy2(src, dst)

    # 2. Brief pause to let Windows breathe
    time.sleep(0.1)

    # 3. Delete the source
    try:
        os.remove(src)
    except PermissionError:
        # If it fails, wait a second and try one last time
        time.sleep(1)
        os.remove(src)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 main.py <src_folder> <dest_folder> <model_path>")
        sys.exit(1)

    src = sys.argv[1].removeprefix("[").removesuffix("]")
    if not os.path.exists(src):
        print("Source folder does not exist.")
        sys.exit(1)

    dest = sys.argv[2].removeprefix("[").removesuffix("]")
    if not os.path.exists(dest):
        os.makedirs(dest)

    model = sys.argv[3].removeprefix("[").removesuffix("]")
    pipeline = AISorterPipeline(
        reference_dir=None,
        category_model_path=model,
    )


    files = list(os.listdir(src))
    for file in files:
        filePath = os.path.join(src, file)
        ext = os.path.splitext(file)[1].lower()
        
        creation_year = datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).year

        # Try to process the file, but move it regardless of processing success
        try:
            if ext in IMAGE_EXTENSIONS:
                try:
                    pipeline.process_image(filePath)
                except Exception as e:
                    print(f"Error processing image {file}: {e}")

            elif ext in VIDEO_EXTENSIONS:
                all_categories = set()
                all_identities = set()

                with tempfile.TemporaryDirectory() as temp_dir:
                    base_name = os.path.splitext(file)[0]
                    saved_frames = extract_frames_from_video(filePath, temp_dir, base_name)

                    for frame_path in saved_frames:
                        try:
                            # process_image evaluates categories and identifies faces in the frame
                            res = pipeline.process_image(frame_path)
                            all_categories.update(res["categories"])
                            all_identities.update(res["identities"])
                        except Exception as e:
                            print(f"Error processing frame {frame_path}: {e}")

                labels = list(all_identities) + [c for c in all_categories if c not in all_identities]
                try:
                    print("Labels:", labels)
                    write_keywords(filePath, labels)
                except Exception as e:
                    print(f"Error writing keywords to video {file}: {e}")
                    
        except Exception as e:
            print(f"Unexpected error processing {file}: {e}")

        # Always route and move the original file, regardless of processing success/failure
        folder = "Photos from " + str(creation_year)
        dest_folder = os.path.join(dest, folder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
            print(f"Created Folder {dest_folder}")

        move_file(filePath, os.path.join(dest_folder, file))

    sys.exit(0)
