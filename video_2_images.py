import cv2 


cap = cv2.VideoCapture('output.webm')
frame_num = 0

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_num % 5 == 0:
            cv2.imwrite('test/' + str(frame_num) + '.png', gray)
    else:
        break
    frame_num +=1

cap.release()
    

