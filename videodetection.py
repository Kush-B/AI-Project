from webbrowser import get
import keras
import tensorflow
import cv2
import os
from deepface import DeepFace
import numpy as np  # this will be used later in the process


cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()

    result = DeepFace.analyze(frame)
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    font = cv2.FONT_HERSHEY_COMPLEX
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, result, (25, 25), font,
                    1, (0, 255, 0), 2, cv2.LINE_4)
        cv2.imshow("video", frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
