import os
import cv2
from pathlib import Path
from PIL import Image
import numpy
import pickle
import sys

def recognise_face():
    labels = {}
    with open('labels.pickle','rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    vid_cam = cv2.VideoCapture(0)
    trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    try:
    	recogniser.read("trainer.yml")
    except:
    	print("Please Train Your Model First")

    while (True):
        startframe, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        face_cordinates = trained_data.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in face_cordinates:
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (0,255,0), 2)
            roi = gray[y:y+h, x:x+w]
            id_, conf = recogniser.predict(roi)
            if conf <= 50:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_frame, labels[id_], (x,y), font, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('frame',image_frame)
        key = cv2.waitKey(1)
        if key == 113:
            exit()
            break
