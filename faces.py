# Import necessary libraries
import numpy as np 
import cv2
import pickle

labels = {}

with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
    # labels = {v:k for k,v in labels.items()}
# Assign the video feed to the variable cap
# The 0 as the argument refers to the default camera
cap = cv2.VideoCapture(0)

# This is the loading the face_cascade, make sure the path to .xml file is correct
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() # pylint: disable=no-member
recognizer.read('trainer.yml')
while(True):
    # ret will be true as long as the video is connected 
    # frame is literally the frame at the current time
    ret, frame = cap.read()

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to gray because it is needed to use the haar cascade 

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # iterable of tuples (x,y,w,h) of detected faces in the frame

    # grab the ROI (Region of interest) from faces
    for (x,y,w,h) in faces:

        # Because it goes y,x (aka rows and columns)
        # The origin is in the top left
        roi_gray = gray[y:y+h, x:x+w]

        roi_color = frame[y:y+h, x:x+w]
        
        id_,conf = recognizer.predict(roi_gray)
        if conf >=45 and conf<=85:
            print(labels[id_])


        img_item = 'my-image.png' # Setting the path of the image to be generated

        # Write to the path you gave and the image is roi_gray (the face in last frame taken because cv2.imwrite just overwrites over and over)
        cv2.imwrite(img_item, roi_gray)


        #Drawing Rectangle
        color = (255,0,0) #BGR because opencv is weird, 0-255
        stroke = 2 #Thickness of line
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke) # cv2.rectangle(frame, (start_x, start_y  of top left), (end_x, end_y of bottom right) )

    # show the frame in a window that launches called "frame"
    cv2.imshow('frame', frame)

    # Wait 20 milliseconds, then if the user presses q, break out of the while loop
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# releases the video stream from cap
cap.release()

# closes the windows that were generated from opencv
cv2.destroyAllWindows()