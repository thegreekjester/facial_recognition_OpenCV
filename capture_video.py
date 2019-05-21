import cv2 

cap = cv2.VideoCapture(0) #reading in from default camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+ 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+ 0.5)
fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps)


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4',fourcc, 15, (width,height)) #very important that its width THEN height

while (True):

    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
        cv2.imshow('gray', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'): #if i press 'q' quit the stream
            break
    
    else:
        break


# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
