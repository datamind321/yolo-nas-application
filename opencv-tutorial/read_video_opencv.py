import cv2 

cap=cv2.VideoCapture('videos/car.mp4')

if (cap.isOpened()==False):
    print("Video Opening Error !!")
while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        resize_frame = cv2.resize(frame,(1280,720))
        cv2.imshow('frame',resize_frame)
        if cv2.waitKey(12) & 0xff == ord('q'):
            break 
    else:
        break 
cap.release()
cv2.destroyAllWindows()   