import numpy as np
import cv2
import os
import face_recognition as fr

print(fr)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Give path of where training_data.yml is saved
face_recognizer.read(r'training_data.yml')

# Replace 0 with "video path" if you want to recognise face from a video
cap = cv2.VideoCapture(0)

name = {0: "Subha"}  # Namkaran of the labels

while True:
    ret, test_img = cap.read()
    # Detecting the face.
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("face Detected: ", faces_detected)

    for (x, y, w, h) in faces_detected:
        # Draws bounding boxes.
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    for face in faces_detected:
        # faces is a list having x,y coordinates and width and height of the img.
        (x, y, w, h) = face
        # Refer: https://stackoverflow.com/questions/57068928/opencv-rect-conventions-what-is-x-y-width-height  to know about conventions.
        roi_gray = gray_img[y:y+h, x:x+h]

        # face_recogniser.predict() returns a tuple having label and confidence score
        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence :", confidence)
        print("label :", label)

        fr.draw_rect(test_img, face)  # Draws bounding boxes
        # Extracting value from the dictionary "name"
        predicted_name = name[label]
        fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (700, 700))

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(10) == ord('q'):
        break
