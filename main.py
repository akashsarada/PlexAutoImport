import datetime
import os


def move_file(src, dest):
    """Move file from src to dest."""
    if os.path.exists(src):
        if os.path.exists(dest):
            print("Destination file already exists. Please choose another destination.")
        else:
            os.rename(src, dest)
    else:
        print("Source file does not exist. Please choose another source.")


if __name__ == '__main__':
    src = input("Enter source folder: ")
    if not os.path.exists(src):
        print("Source folder does not exist. Please choose another source.")
        exit(1)

    dest = input("Enter destination folder: ")
    if not os.path.exists(dest):
        os.mkdir(dest)

    for file in os.listdir(src):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(
                ".gif") or file.endswith(".mp4"):
            # get file name using os
            creation_year = datetime.datetime.fromtimestamp(os.path.getmtime(file)).year

            folder = "Photos from" + str(creation_year)
            if os.path.exists(os.path.join(dest, folder)):
                move_file(os.path.join(src, file), os.path.join(dest, folder, file))
            else:
                os.mkdir("Photos from" + creation_year)
                move_file(os.path.join(src, file), os.path.join(dest, folder, file))
    exit(0)
