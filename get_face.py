import cv2
import os
import sys

def get_face():
    # make directory to store students face
    face_id=input('Enter Name :  ')
    vid_cam = cv2.VideoCapture(0)
    trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    dir = (f'students/{face_id}/')
    if not os.path.exists(dir):
        os.makedirs(dir)
    while(True):
        startframe, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        face_cordinates = trained_data.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in face_cordinates:
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (0,255,0), 2)
            count += 1
            cv2.imwrite(f'students/{face_id}/{face_id}' + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('frame', image_frame)
        key = cv2.waitKey(100)
        if key == 113:
            con.close()
            break
        elif count>=50:
            print("Successfully Captured")
            break
