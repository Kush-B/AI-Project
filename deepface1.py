# emotion_detection.py
from webbrowser import get
import keras
import tensorflow
import cv2
import os
from deepface import DeepFace
import numpy as np  # this will be used later in the process


# put the image where this file is located and put its name here
imgpath = 'humans/1 (477).jpg'
image = cv2.imread(imgpath)
# ---------------------------------------------------
# for checking the image size:
dimensions = image.shape

# height, width, number of channels in image
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]
dim = (width, height)
# print('Image Dimension    : ', dimensions)
# print('Image Height       : ', height)
# print('Image Width        : ', width)
# print('Number of Channels : ', channels)
# ---------------------------------------------------

# For resizing
# img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)

# print('Original Dimensions : ',img.shape)
if height > 1080 and width > 920:
    scale_percent = 50  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)


# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# --------------------------------------------------------------------

# obj = DeepFace.analyze(img_path="humans/1 (8).jpg",actions=['age', 'gender', 'race', 'emotion'])
# OR
predictions = DeepFace.analyze(resized)
print(predictions)
# dominant_race,gender,dominant_emotion
# face detection rectangle plus the detection outputs
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
font = cv2.FONT_HERSHEY_COMPLEX
# to retrive multiple values from keys

# keys = ['gender', 'dominant_race', 'dominant_emotion']
# values = list(map(predictions.get, keys))

# text = predictions.get('dominant_emotion', 'gender')
# print(values)
for(x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(resized, predictions['dominant_emotion'],
                (25, 25), font, 1, (0, 255, 0), 2, cv2.LINE_4)

cv2.imshow("image", resized)

# print("printing the result")
# print(obj)
# cv2.imshow("image", image)
cv2.waitKey(0)
# print(tensorflow.__version__)
# print(keras.__version__)

# here the first parameter is the image we want to analyze #the second one there is the action
# analyze = DeepFace.analyze(image, actions=['emotions'])
# print(analyze)
