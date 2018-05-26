import cv2
from lib import lib
from lib import camthread

stream = camthread.VideoStream(src="http://10.9.0.120:8080/video").start()
try:
    while True:
        image = stream.read()
        if image is None:
            continue

        face, rect, all_faces = lib.detect_faces(image)

        if rect is None:
            title = "No face detected"
        else:
            for r in all_faces:
                (x, y, w, h) = r
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            title = "Face detected"

        cv2.putText(image, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if face is not None:
            cv2.imshow("face", face)

        cv2.imshow("preview", image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cv2.waitKey(0)
finally:
    stream.stop()
    cv2.destroyAllWindows()
