import cv2
import os
from keras.utils.image_utils import img_to_array
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model("Mask_Detection.model")

def face_mask_detector(frame):
    # frame = cv2.imread(fileName)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list=[]
    preds=[]
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)
        faces_list.append(face_frame)
        if len(faces_list)>0:
            preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y- 10),cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)
    return frame


input_image = cv2.imread("D:\Study\Projects\Mask Detection\Test\modiji.jpg") #Image Location
image = cv2.resize(input_image,(500,500))
output = face_mask_detector(image)
cv2.imshow('Mask Detection',output)
cv2.waitKey(0)