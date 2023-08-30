import cv2

img = cv2.imread('images/night.jpg')
resize_img=cv2.resize(img,(1280,720))
cv2.imshow('img',resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()