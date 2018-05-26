import cv2

g_face_classifier = None


def get_face_classifier():
    global g_face_classifier

    if g_face_classifier is not None:
        return g_face_classifier

    # LBP face detector is faster but less accurate
    # g_face_classifier = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    # Haar classifier is slower but more accurate
    g_face_classifier = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    return g_face_classifier


# function to detect face using OpenCV
def detect_faces(img, return_colour=False):
    # convert image to grey for the face detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = get_face_classifier().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None, []

    # Get the first face and return the image and rect
    (x, y, w, h) = faces[0]
    if return_colour:
        return img[y: y + w, x: x + h], faces[0], faces
    return gray[y: y + w, x: x + h], faces[0], faces
