# Import necessary libraries
import numpy as np 
import cv2
import pickle

labels = {}
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') # pylint: disable=no-member

with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
    # labels = {v:k for k,v in labels.items()}
# Assign the video feed to the variable cap
# The 0 as the argument refers to the default camera
cap = cv2.VideoCapture(0)

# This loads the recognizer with info (trainer.yml) about the training
recognizer = cv2.face.LBPHFaceRecognizer_create() # pylint: disable=no-member
recognizer.read('trainer_DNN.yml')

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        [h, w] = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), # pylint: disable=no-member
        1.0, (300, 300), (104.0, 177.0, 123.0)) 
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < .5:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[startY:endY, startX:endX]
            if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                id_, uncertainty = recognizer.predict(roi_gray)
                print(uncertainty)
            if uncertainty <60:
                name = labels[id_]
                #Drawing Rectangle
                color = (255,0,0) #BGR because opencv is weird, 0-255
                stroke = 2 #Thickness of line
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, name, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    # show the output frame
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    else:
        break


# releases the video stream from cap
cap.release()
# closes the windows that were generated from opencv
cv2.destroyAllWindows()