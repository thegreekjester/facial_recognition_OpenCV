import numpy as np 
import cv2 
import pickle 


def predict_image(img_path, pickle_path, yml_path):
    """
    Keyword arguments:
    img_path -- path to the color image of a single person
    pickle_path -- path to the labels pickle file
    yml_path -- path to the yml file to use for the recognizer

    this function takes a color image of one person and returns the 
    facial classification value as a string
    """
    labels = {}
    # the neural net loaded from a pre-trained caffe model
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel') # pylint: disable=no-member

    # oepn the pickle file and load them into the labels dictionary previously created
    with open(pickle_path, 'rb') as f:
        labels = pickle.load(f)

    # read in image as grayscale because it is needed for facial recognition
    img = cv2.imread(img_path)

    # grab height and width from the image shape (h, w, color_channels)
    [h,w] = img.shape[:2]

    # This loads the recognizer with info (trainer.yml) about the training
    recognizer = cv2.face.LBPHFaceRecognizer_create() # pylint: disable=no-member
    recognizer.read(yml_path)

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), # pylint: disable=no-member
        1.0, (300, 300), (104.0, 177.0, 123.0)) 
    net.setInput(blob)
    detections = net.forward()
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # this roi_gray is just region of the image that has been identified as a face
            roi_gray = gray[startY:endY, startX:endX]
            # As long as the roi_gray is bigger than 0, predict the facial classification
            if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                id_, uncertainty = recognizer.predict(roi_gray)
                print(uncertainty)
            # if the uncertainty of the classification is less than 60, print the name value
            if uncertainty <55:
                name = labels[id_]
                print(name)
                return name
    print('nothing found')
    return None