import numpy as np
import cv2
import os


def faceDetection(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    face_haar = cv2.CascadeClassifier(
        r"C:\Users\ASUS\anaconda3\envs\facerecognition\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
    faces = face_haar.detectMultiScale(
        gray_image, scaleFactor=1.3, minNeighbors=3)

    return faces, gray_image


def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print("Skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)

            print("img_path", img_path)
            print('id', id)

            test_img = cv2.imread(img_path)
            if test_img is None:
                print('Not loaded properply')
                continue

            faces_rect, gray_img = faceDetection(test_img)
            # faces is a list having x,y coordinates and width and height of the img.Refer: https://stackoverflow.com/questions/57068928/opencv-rect-conventions-what-is-x-y-width-height  to know about conventions
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y + h, x:x + w]  # roi == region of interest
            faces.append(roi_gray)
            faceID.append(int(id))

    return faces, faceID


def trainClassifier(faces, faceID):
    """Trains the model"""
    face_recogniser = cv2.face.LBPHFaceRecognizer_create()
    face_recogniser.train(faces, np.array(faceID))

    return face_recogniser


def draw_rect(test_img, face):
    """Draws bounding boxes"""
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0),
                  thickness=2)  # Draws bounding boxes


def put_text(test_img, label_name, x, y):
    """Puts name on the top of the bounding box"""
    cv2.putText(test_img, label_name, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 2)


def train():
    test_img = cv2.imread(r"img.jpg")

    faces_detected, gray_img = faceDetection(test_img)

    # Training will begin from here

    faces, faceID = labels_for_training_data(
        r'Images')
    face_recogniser = trainClassifier(faces, faceID)
    face_recogniser.save(
        r'trainingData.yml')

    name = {0: 'Subha'}

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + w]
        label, confidence = face_recogniser.predict(roi_gray)
        print("Label:", label)
        print('Confidence', confidence)
        draw_rect(test_img, face)
        predict_name = name[label]
        put_text(test_img, predict_name, x, y)

    resized_img = cv2.resize(test_img, (700, 700))

    cv2.imshow("Face detection", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    train()
