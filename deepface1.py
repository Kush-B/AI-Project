# emotion_detection.py
import keras
import tensorflow
import cv2
from deepface import DeepFace
import numpy as np  # this will be used later in the process

# put the image where this file is located and put its name here
imgpath = 'data/frame37.jpg'
image = cv2.imread(imgpath)

obj = DeepFace.analyze(img_path="data/frame37.jpg",
                       actions=['age', 'gender', 'race', 'emotion'])
print("printing the result")
print(obj)
cv2.imshow("image", image)
# print(tensorflow.__version__)
# print(keras.__version__)

# here the first parameter is the image we want to analyze #the second one there is the action
# analyze = DeepFace.analyze(image, actions=['emotions'])
# print(analyze)
