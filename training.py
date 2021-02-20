import os
import cv2
from pathlib import Path
from PIL import Image
import numpy
import pickle

def training_img():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR,"students")

    trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recogniser = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root,file)
                label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id +=  1
                id_ = label_ids[label]
                pil_image = Image.open(path)
                image_array = numpy.array(pil_image,"uint8")
                faces = trained_data.detectMultiScale(image_array,1.3,5)
                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open('labels.pickle','wb') as f:
        pickle.dump(label_ids, f)

    recogniser.train(x_train, numpy.array(y_labels))
    recogniser.save("trainer.yml")
    print("Trained succesfully")
