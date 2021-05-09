import numpy as np
import cv2
import os
import face_recognition as fr


test_img = cv2.imread(r"img.jpg")  # path to img


faces_detected, gray_img = fr.faceDetection(test_img)
print('Face detected')


face_recogniser = cv2.face.LBPHFaceRecognizer_create(12)

# Reads data from the yml file
face_recogniser.read(r'trainingData.yml')  # Path to the yml file

name = {0: "Subha"}  # Namkaran of the labels

for face in faces_detected:
    # faces is a list having x,y coordinates and width and height of the img.
    (x, y, w, h) = face
    # Refer: https://stackoverflow.com/questions/57068928/opencv-rect-conventions-what-is-x-y-width-height  to know about conventions
    roi_gray = gray_img[y:y + h, x:x + w]
    # face_recogniser.predict() returns a tuple having label and confidence score
    label, confidence = face_recogniser.predict(roi_gray)

    print('Confidence =', str(confidence) + "%")  # The lower, the better
    print("Label:", label)

    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (700, 700))


cv2.imshow("Face Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
