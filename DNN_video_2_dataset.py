import cv2 
import numpy as np
import argparse

# Reading in the Caffe Model for Face Detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') # pylint: disable=no-member
# target the webcam
cap = cv2.VideoCapture(0)

frame_num = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--person", type=str, required=True, help="person that this is tracking")
args = vars(ap.parse_args())

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        # frame is (h,w,color_channels)
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
    
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            	# show the output frame
            if frame_num % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_roi = gray[startY:endY, startX:endX]
                print('dataset/' + args['person'] + '/' + str(frame_num) + '.png')
                cv2.imwrite('dataset/' + args['person'] + '/' + str(frame_num) + '.png', gray_roi)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        frame_num+=1
    else:
        break
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()