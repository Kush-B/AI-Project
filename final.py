from deepface import DeepFace
import cv2
# file for extracting images from the video
# Importing all necessary libraries
from asyncio.windows_events import NULL
import cv2
import os


# cam = cv2.VideoCapture(
#     'C:/Users/bhard/OneDrive/Desktop/AI/Project/AI-Project/1.mp4')
imgPath = 'C:/Users/bhard/OneDrive/Desktop/AI/Project/AI-Project/1.mp4'
cap = cv2.VideoCapture(imgPath)


def videoExractor():

    try:

        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0
    counter = 0
    while(True):

        # reading from frame
        ret, frame = cap.read()

        if ret and counter % 30 == 0:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)
            print(counter)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        elif (ret == NULL):
            cap.release()
            cv2.destroyAllWindows()
        counter += 1


def faceVerification():
    # models to test
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace",
              "DeepFace", "DeepID", "ArcFace", "SFace"]

    # Verify the image with test image (IDs) with different models:
    result1 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[0])

    result2 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[1])

    result3 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[2])
    result4 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[3])
    result5 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[4])

    result6 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[5])
    result7 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[6])

    result8 = DeepFace.verify(img1_path="C:/Users/bhard/Downloads/person 1.jpg",
                              img2_path="C:/Users/bhard/Downloads/person1Test.jpg", model_name=models[7])

    print(" result1['distance'] ", result1['distance'], "------",
          "result1['verified']", result1['verified'], "\n"),
    print(" result2['distance'] ", result2['distance'], "------",
          "result2['verified']", result2['verified'], "\n"),
    print(" result3['distance'] ", result3['distance'], "------",
          "result3['verified']", result3['verified'], "\n"),
    print(" result4['distance'] ", result4['distance'], "------",
          "result4['verified']", result4['verified'], "\n"),
    print(" result5['distance'] ", result5['distance'], "------",
          "result5['verified']", result5['verified'], "\n"),
    print(" result6['distance'] ", result6['distance'], "------",
          "result6['verified']", result6['verified'], "\n"),
    print(" result7['distance'] ", result7['distance'], "------",
          "result7['verified']", result7['verified'], "\n"),
    print(" result8['distance'] ", result8['distance'], "------",
          "result8['verified']", result8['verified'], "\n")

    # To write the result of findings in a file
    with open('filename.txt', 'w') as f:
        f.write(str(result1['distance']))

    # To find the image in a folder of IDs
    df = DeepFace.find(
        img_path="C:/Users/bhard/Downloads/person 1.jpg", db_path="C:/Users/bhard/OneDrive/Desktop/AI/Project/data", model_name=models[4])
    print(df['identity'])

    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open!")

    # frontal face cascades

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # eye_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + 'haarcascade_eye.xml')
    # profile_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + 'haarcascade_profileface.xml')

    counter = 0
    while True:
        # reading video frame by frame
        ret, frame = cap.read()

        # resizing frame
        resize = cv2.resize(frame, (640, 480))
        # result = DeepFace.analyze(resize, actions=['emotion'])
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        # detecting face
        faces = faceCascade.detectMultiScale(gray, 1.3, 3)

        # draw a rectangle around the faces:

        for(x, y, w, h) in faces:
            # print(x, y, w, h)
            result = DeepFace.analyze(resize, actions=['emotion'])
            # print(result['dominant_emotion'])
            # roi_gray = gray[y:y+h, x:x+w]  # (ychord_start, ychord_end)
            # roi_color = resize[y:y+h, x:x+w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            position = cv2.rectangle(
                resize, (x, y), (x+w, y+h), (0, 255, 0), 2)

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
        if(counter % 60 == 0):
            {
                print("----------------\n"),
                # print(x, y),
                print(result['dominant_emotion']),
                print("----------------\n")
            }
        counter = counter + 1
        # printEmotion()
        cv2.imshow('Demo', resize)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


faceVerification()
videoExractor()
