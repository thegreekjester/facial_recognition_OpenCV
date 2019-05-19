import os # used to interact with files on the computer
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle 
import numpy as np



# This is the loading the face_cascade, make sure the path to .xml file is correct
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create() # pylint: disable=no-member

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # grabs the absolute path of this file's directory

IMAGE_DIR = os.path.join(BASE_DIR, 'images') # take BASE_DIR and grabs a reference to the images folder within it

x_train = []
y_train = []
# Looping through the files in the image_dir
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files: #for each file
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg') or file.endswith('JPG'): #if it ends with .png or .jpg
            path = os.path.join(root, file) #print its path
            label = os.path.basename(root).replace(' ', '_').lower() # returns the directory that the image is in (to be used as a label)

            y_train.append(label) # put the label (dir name) into y train
            img = cv2.imread(path, 0)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=7)
            if len(faces) == 0:
                print(path)
            elif len(faces) > 1:
                print('multiple {}'.format(faces))
                faces = [faces[1]]
            else:
                print('just one: {}'.format(path))
            for (x,y,w,h) in faces:
                roi = img[y:y+h, x:x+w]
                x_train.append(roi)
            

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_train_labels = label_encoder.inverse_transform(y_train)

label_ids = dict(zip(y_train, y_train_labels))

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, y_train)
recognizer.save('trainer.yml')
