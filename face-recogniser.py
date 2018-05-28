import cv2
import os
import numpy as np
import time
from lib import lib
from lib import camthread


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    print("Preparing data...")

    # list of face rects
    faces = []
    # list of face labels
    labels = []
    # Dictionary of label => name
    label_names = {}
    # Current label ID
    label = 0

    dirs = os.listdir(data_folder_path)
    for dir_name in dirs:
        # Take the next label number
        label = label + 1
        label_names[label] = dir_name

        subject_dir = data_folder_path + "/" + dir_name
        if not os.path.isdir(subject_dir):
            continue

        # get the images names that are inside the given subject directory
        image_files = os.listdir(subject_dir)

        for file_name in image_files:
            full_path = subject_dir + "/" + file_name
            if file_name.startswith(".") or not os.path.isfile(full_path):
                continue

            image = cv2.imread(full_path)

            # cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            # cv2.waitKey(10)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces.append(gray)
            labels.append(label)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    print("Data prepared. Total faces:", len(faces), " Total labels:", len(label_names))

    return faces, labels, label_names


def save_unknown_face(img, save_dir, label_name, confidence):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = "%s/%s_%s-%0.1f.jpg" % (save_dir, time.strftime("%Y%m%d%H%M%S"), label_name, confidence)
    cv2.imwrite(filename, img)


def main():
    # lower means stricter
    max_confidence = 150
    unknown_faces_dir = "unknown_faces"
    training_dir = "training-data"
    source_url = "http://10.9.0.120:8080/video"

    # Initialise and train face recogniser
    faces, labels, label_names = prepare_training_data(training_dir)
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    # recogniser = cv2.face.EigenFaceRecognizer_create()
    # recogniser = cv2.face.FisherFaceRecognizer_create()
    print("Training face recognizer...")
    recogniser.train(faces, np.array(labels))
    print("Training finished")

    stream = camthread.VideoStream(src=source_url).start()
    try:
        quit_pressed = False
        while not quit_pressed:
            image = stream.read()

            if image is None:
                continue

            face, rect, all_faces = lib.detect_faces(image)

            if rect is not None:
                # Classify the face
                label, confidence = recogniser.predict(face)
                (x, y, w, h) = rect

                if confidence <= max_confidence:
                    face_name = label_names[label]
                else:
                    face_name = "Unknown (%s)" % label_names[label]
                    save_unknown_face(image[y: y + w, x: x + h], unknown_faces_dir, label_names[label], confidence)

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, "%s (%f)" % (face_name, confidence), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)

                if face is not None:
                    cv2.putText(face, face_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("face", face)

            cv2.imshow("preview", image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                quit_pressed = True

        if not quit_pressed:
            cv2.waitKey(0)
    finally:
        stream.stop()
        cv2.destroyAllWindows()


main()
