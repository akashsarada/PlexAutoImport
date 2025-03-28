import datetime
import os
import shutil
import sys


def move_file(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            print("Destination file already exists. Please choose another destination.")
        else:
            print("Moving file " + str(src) + " to " + str(dest))
            shutil.move(src, dest)
    else:
        print("Source file does not exist. Please choose another source.")


if __name__ == '__main__':
    src = sys.argv[1].removeprefix("[").removesuffix("]")
    if not os.path.exists(src):
        print("Source folder does not exist. Please choose another source.")
        exit(1)

    dest = sys.argv[2].removeprefix("[").removesuffix("]")
    if not os.path.exists(dest):
        os.makedirs(dest)

    for file in os.listdir(src):
        filePath = os.path.join(src, file)

        if filePath.endswith(".jpg") or filePath.endswith(".png") or filePath.endswith(".jpeg") or filePath.endswith(".gif") or filePath.endswith(".mp4") or filePath.endswith(".heic") or filePath.endswith(".dng"):
            creation_year = datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).year

            folder = "Photos from " + str(creation_year)
            if os.path.exists(os.path.join(dest, folder)):
                move_file(os.path.join(src, file), os.path.join(dest, folder, file))
            else:
                os.makedirs(os.path.join(dest, folder))
                print("Created Folder " + os.path.join(dest, folder))
                move_file(os.path.join(src, file), os.path.join(dest, folder, file))
    exit(0)
