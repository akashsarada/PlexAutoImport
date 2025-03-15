import os
import sys
from datetime import datetime

import main

photos = []

if __name__ == '__main__':
    dir = sys.argv[1].removeprefix("[").removesuffix("]")
    threshold = int(sys.argv[2].removeprefix("[").removesuffix("]"))
    for file in os.listdir(dir):
        photos.append(file)
        if len(photos) > threshold:
            print("Last " + str(threshold) + " files: " + str(photos))
            f_date = photos[0].split("_")[0]
            l_date = photos[-1].split("_")[0]
            print("First Date: " + f_date + ", Last Date: " + l_date)
            if f_date == l_date:
                folder = "Event on " + str(f_date)
                if not os.path.exists(os.path.join(dir, folder)):
                    os.makedirs(os.path.join(dir, folder))
                main.move_file(os.path.join(dir, photos[0]), os.path.join(dir, folder, photos[0]))
            if os.path.exists(os.path.join(dir, "Event on " + f_date)):
                main.move_file(os.path.join(dir, photos[0]), os.path.join(dir, "Event on " + f_date, photos[0]))
            photos.pop(0)