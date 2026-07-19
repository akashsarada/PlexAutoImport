import datetime
import os
import shutil
import sys
import time

def move_file(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            print("Destination file already exists. Please choose another destination.")
        else:
            print("Moving file " + str(src) + " to " + str(dest))
            robust_move(src, dest)
    else:
        print("Source file does not exist. Please choose another source.")

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

if __name__ == '__main__':
    src = sys.argv[1].removeprefix("[").removesuffix("]")
    if not os.path.exists(src):
        print("Source folder does not exist. Please choose another source.")
        exit(1)

    dest = sys.argv[2].removeprefix("[").removesuffix("]")
    if not os.path.exists(dest):
        os.makedirs(dest)

    files = list(os.listdir(src))
    for file in files:
        filePath = os.path.join(src, file)

        extensions = ["jpg", ".png", ".jpeg", ".gif", "mp4", ".heic", ".dng"]
        if os.path.splitext(filePath) in extensions:
            creation_year = datetime.datetime.fromtimestamp(os.path.getmtime(filePath)).year

            folder = "Photos from " + str(creation_year)
            if os.path.exists(os.path.join(dest, folder)):
                move_file(os.path.join(src, file), os.path.join(dest, folder, file))
            else:
                os.makedirs(os.path.join(dest, folder))
                print("Created Folder " + os.path.join(dest, folder))
                move_file(os.path.join(src, file), os.path.join(dest, folder, file))
