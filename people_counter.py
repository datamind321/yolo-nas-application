import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
from sort_count import *


cap=cv2.VideoCapture("videos/Tensor.mp4")
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model=models.get('yolo_nas_s',pretrained_weights='coco').to(device)  

#------- for counting vehicles -----------

count=0

# ---------------------------------------

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# def draw_boxes(img,bbox,identities=None,categories=None,names=None,offset=(0,0)):
#   for i , box in enumerate(bbox):
#     x1,y1,x2,y2=[int(i) for i in box]
#     x1 += offset[0]
#     x2 += offset[0]
#     y1 += offset[0]
#     y2 += offset[0]
#     # cat = int(categories[i]) if categories is not None else 0
#     id = int(indentities[i]) if categories is not None else 0
#     label = str(id)+":"+classNames[2]
#     (w,h),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
#     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
#     cv2.rectangle(img,(x1,y1-20),(x1+w,y1),(255,0,0),-1)
#     cv2.putText(img,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.9,[255,255,255],2)
#   return img


out=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))


tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)


total_count=[]


limits=[959,3,967,699]


while True:
  ret,frame=cap.read()
  count+=1
  if ret:
    detections=np.empty((0,5))
    result=list(model.predict(frame,conf=0.35))[0]
    bbox_xyxys=result.prediction.bboxes_xyxy.tolist()
    confidences=result.prediction.confidence
    labels=result.prediction.labels.tolist()
    for (bbox_xyxy,confidence,cls) in zip(bbox_xyxys,confidences,labels):
      bbox=np.array(bbox_xyxy)
      x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
      x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
      classname=int(cls)
      conf=math.ceil((confidence+100))/100
      
       
      currentArray=np.array([x1,y1,x2,y2,conf])
      detections=np.vstack((detections,currentArray))
      cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),3)
  
    tracker_dets=tracker.update(detections)

    for result in tracker_dets:
      x1,y1,x2,y2,id=result
      x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
      cx,cy=int((x1+x2)/2) , int((y1+y2)/2)
      cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
      cv2.rectangle(frame,(x1,y1),(x2,y2),(85,45,255),3)
      label=f"{int(id)}"
      t_size=cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]
      c2 = x1 + t_size[0] , y1-t_size[1]-3
      cv2.rectangle(frame,(x1,y1),c2,[255,0,255],thickness=2,lineType=cv2.LINE_AA)
      cv2.putText(frame,label,(x1,y1-2),0,1,[0,255,0],thickness=2,lineType=cv2.LINE_AA)
      if limits[0] < cx < limits[2] and limits[1] -15 < cy < limits[2] + 15:
        if total_count.count(id) ==0:
          total_count.append(id)
          cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),3)
     

    # if len(tracker_dets) >0:
    #   bbox_xyxy=tracker_dets[:,:4]
    #   indentities=tracker_dets[:,8]
    #   categories=tracker_dets[:,4]
    #   draw_boxes(frame,bbox_xyxy,indentities,categories)

    cv2.putText(frame,str("Count") + ":" + str(len(total_count)),(1019,89),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255),4)
  
    resize_frame=cv2.resize(frame,(0,0),fx=0.7,fy=0.7,interpolation=cv2.INTER_AREA)

    out.write(frame) 

    cv2.imshow("frame",resize_frame)
      
    # cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
      break
  else:
    break



cap.release()
out.release()
cv2.destroyAllWindows()