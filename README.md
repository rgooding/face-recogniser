
Based very heavily on the excellent tutorial by Ramiz Raja at https://www.superdatascience.com/opencv-face-recognition/
and https://github.com/informramiz/opencv-face-recognition-python

**video-face-detect.py**

Does simple real-time face detection on a video or live stream

**face-recogniser.py**

Performs face detection and recognition on a video or live stream

Put images of faces in subdirectories of training-data/, one subdirectory per person. The classifier will be trained from these images when the script starts.
While the script is running it will place images of unrecognised faces into an unknown_faces directory. You can then use these as additional training data by moving them to training-data/.
