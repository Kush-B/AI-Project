import threading
from deepface import DeepFace
import cv2

# imgPath = 'C:/Users/bhard/OneDrive/Desktop/AI/Project/AI-Project/6.mp4'
# cap = cv2.VideoCapture(imgPath)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open!")
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# eye_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_eye.xml')
# profile_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_profileface.xml')


while True:
    isRunning = False
    if(isRunning):
        continue

    ret, frame = cap.read()
    resize = cv2.resize(frame, (640, 480))
    # result = DeepFace.analyze(resize, actions=['emotion'])
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.5, 3)

    # draw a rectangle around the faces:

    for(x, y, w, h) in faces:
        # print(x, y, w, h)
        result = DeepFace.analyze(resize, actions=['emotion'])
        # print(result['dominant_emotion'])
        # roi_gray = gray[y:y+h, x:x+w]  # (ychord_start, ychord_end)
        # roi_color = resize[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        position = cv2.rectangle(resize, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # profile = profile_cascade.detectMultiScale(roi_gray)
        pposition = cv2.rectangle(
            resize, (x, y-50), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(resize, result['dominant_emotion'],
                    (x+5, y-20),
                    font, 1,
                    (0, 255, 0),
                    2, cv2.LINE_4)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # for (ex, ey, ew, eh) in profile:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # use putText() for inserting text on video
    def printEmotion(self):

        print("////////////////")
        print(position)
        print('\n'+result['dominant_emotion'])
        print("////////////////")
        self.isRunning = False

    isRunning = True
    threading.Timer(10, printEmotion).start()
    # printEmotion()
    cv2.imshow('Demo', resize)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
