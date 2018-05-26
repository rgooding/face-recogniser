import os
import cv2
from lib import lib

images_dir = "images"
faces_dir = "images/faces"


def main():
    files = os.listdir(images_dir)

    for file_name in files:
        full_path = images_dir + "/" + file_name

        if not os.path.isfile(full_path):
            continue

        print("Processing " + full_path)

        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        _, _, faces = lib.detect_faces(img, return_colour=True)
        n = 1
        for face in faces:
            (x, y, w, h) = face
            imgfile = "%s/%s__%d.jpg" % (faces_dir, file_name, n)
            # imgfile = faces_dir + "/" + file_name + "__" + n + ".jpg"
            cv2.imwrite(imgfile, img[y: y + w, x: x + h])


main()
