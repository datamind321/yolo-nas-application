import cv2 

cap=cv2.VideoCapture(0)

if (cap.isOpened==False):
    print("Video Playing error !!!")

while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(24) & 0xff == ord('q'):
            break 
    else:
        break 

cap.release()
cv2.destroyAllWindows()
