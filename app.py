import streamlit as st 
from super_gradients.training import models 
import torch
import numpy as np
import math
import cv2
import tempfile
import time
from PIL import Image
from super_gradients.common.object_names import Models
import av
from streamlit_webrtc import webrtc_streamer,RTCConfiguration

class VideoProcessor:
    def recv(self,frame):
        frm=frame.to_ndarray(format="bgr24")
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
        model = models.get(Models.YOLOX_N,pretrained_weights='coco').to(device)
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
        
        result=list(model.predict(frm,conf=0.35))[0]
        bbox_xyxys=result.prediction.bboxes_xyxy.tolist()
        confidences=result.prediction.confidence
        labels=result.prediction.labels.tolist()
        for (bbox_xyxy,confidence,cls) in zip(bbox_xyxys,confidences,labels):
           bbox=np.array(bbox_xyxy)
           x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
           x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
           classname=int(cls)
           class_name=classNames[classname]
           conf=math.ceil((confidence+100))/100
           label=f"{class_name}{conf}"
           t_size=cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]
           c2 = x1 + t_size[0] , y1-t_size[1]-3
           cv2.rectangle(frm,(x1,y1),(x2,y2),(0,255,255),3)
           cv2.rectangle(frm,(x1,y1),c2,[255,0,255],thickness=1,lineType=cv2.LINE_AA)
           cv2.putText(frm,label,(x1,y1-2),0,1,[0,255,0],thickness=2,lineType=cv2.LINE_AA)
     
        return av.VideoFrame.from_ndarray(frm,format="bgr24")




def load_process_on_img(img,conf,st):
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model=models.get('yolo_nas_s',pretrained_weights='coco').to(device)

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
    result=list(model.predict(img,conf=0.35))[0]
    bbox_xyxys=result.prediction.bboxes_xyxy.tolist()
    confidences=result.prediction.confidence
    labels=result.prediction.labels.tolist()
    for (bbox_xyxy,confidence,cls) in zip(bbox_xyxys,confidences,labels):
      bbox=np.array(bbox_xyxy)
      x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
      x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
      classname=int(cls)
      class_name=classNames[classname]
      conf=math.ceil((confidence+100))/100
      label=f"{class_name}{conf}"
      t_size=cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]
      c2 = x1 + t_size[0] , y1-t_size[1]-3
      cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)
      cv2.rectangle(img,(x1,y1),c2,[255,0,255],thickness=1,lineType=cv2.LINE_AA)
      cv2.putText(img,label,(x1,y1-2),0,1,[0,255,0],thickness=2,lineType=cv2.LINE_AA)
    st.subheader("Output Image")
    st.image(img,channels='RGB',use_column_width=True)

def yolo_nas_process_on_frame(video,kpi_text,kpi2_text,kpi3_text,stframe):
    cap=cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model=models.get('yolo_nas_s',pretrained_weights='coco').to(device)
    stframes=st.empty() 
    count=0
    prev_time=0
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
    
    out=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))
    while True:
       ret,frame=cap.read()
       count+=1
       if ret:
        result=list(model.predict(frame,conf=0.35))[0]
        bbox_xyxys=result.prediction.bboxes_xyxy.tolist()
        confidences=result.prediction.confidence
        labels=result.prediction.labels.tolist()
          
        for (bbox_xyxy,confidence,cls) in zip(bbox_xyxys,confidences,labels):
            bbox=np.array(bbox_xyxy)
            x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            classname=int(cls)
            class_name=classNames[classname]
            conf=math.ceil((confidence+100))/100
            label=f"{class_name}{conf}"
            t_size=cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]
            c2 = x1 + t_size[0] , y1-t_size[1]-3
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),3)
            cv2.rectangle(frame,(x1,y1),c2,[255,0,255],thickness=2,lineType=cv2.LINE_AA)
            cv2.putText(frame,label,(x1,y1-2),0,1,[0,255,0],thickness=2,lineType=cv2.LINE_AA)
        stframe.image(frame,channels="BGR",use_column_width=True)
        current_time=time.time()
        fps=1/(current_time-prev_time)
        prev_time=current_time
        kpi_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>",unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>",unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>",unsafe_allow_html=True)
   
st.title("Yolo NAS Object Detection Streamlit Application")
st.sidebar.title("Setting")
st.sidebar.subheader("Parameters")
st.markdown(
    """ <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child { 
        width:300px;
     }
     [data-testid='stSidebar'][aria-expanded='false'] > div:first-child { 
        width:300px;
        margin-left:-300px;
     }
    </style>"""
,unsafe_allow_html=True)  
app_mode=st.sidebar.selectbox("Choose the app mode",["Run On Image","Run On Video","About App"])


if app_mode=="Run On Image":
    st.sidebar.markdown("---")
    confidence=st.sidebar.slider('Confidences',min_value=0.0,max_value=1.0)
    st.sidebar.markdown("---")
    st.markdown(
    """ <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child { 
        width:300px;
     }
     [data-testid='stSidebar'][aria-expanded='false'] > div:first-child { 
        width:300px;
        margin-left:-300px;
     }
    </style>"""
,unsafe_allow_html=True) 
    image_file=st.sidebar.file_uploader("Upload an Image",type=['jpg','jpeg','png','webp'])
    if image_file is not None:
        img=cv2.imdecode(np.fromstring(image_file.read(),np.uint8),1)
        img=np.array(Image.open(image_file))
        st.sidebar.image(image_file)
        load_process_on_img(img,confidence,st)

if app_mode=="Run On Video":
    st.sidebar.markdown("---")
    confidence=st.sidebar.slider('Confidences',min_value=0.0,max_value=1.0)
    st.sidebar.markdown("---")
    st.markdown(
    """ <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child { 
        width:300px;
     }
     [data-testid='stSidebar'][aria-expanded='false'] > div:first-child { 
        width:300px;
        margin-left:-300px;
     }
    </style>"""
,unsafe_allow_html=True) 
    st.sidebar.markdown("---")   
    use_webcam=st.sidebar.checkbox('Use Webcam')
    st.sidebar.markdown("---")
    video_file=st.sidebar.file_uploader('Upload an Video',type=['mp4','avi','mov','asf'])

    tfile=tempfile.NamedTemporaryFile(suffix='.mp4',delete=False)
    if not video_file:
       if use_webcam:
          webrtc_streamer(key="key",video_processor_factory=VideoProcessor,rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}))
          
          
          

    #    else:
    #       vid=cv2.VideoCapture(video_file)
    #       tfile.name=video_file
    #       video=open(tfile.name,'rb')
    #       video_bytes=video.read()
    #       st.sidebar.text("Input Video")
    #       st.sidebar.video(video_bytes)
    else:
       tfile.write(video_file.read())
       video=open(tfile.name,'rb')
       video_bytes=video.read()
       st.sidebar.text("Input Video")
       st.sidebar.video(video_bytes)
       stframe=st.empty()
       st.markdown("<hr/>",unsafe_allow_html=True)
       kpi1,kpi2,kpi3=st.columns(3)
       with kpi1:
          st.header("Frame Rate")
          kpi_text=st.markdown("0")
       with kpi2:
          st.header("Width")
          kpi2_text=st.markdown("0")
       with kpi3:
         st.header("Height")
         kpi3_text=st.markdown("0")
       st.markdown("<hr/>",unsafe_allow_html=True)
       yolo_nas_process_on_frame(tfile.name,kpi_text,kpi2_text,kpi3_text,stframe)

   

if app_mode=="About App":
    st.image("Deci-Logo.png")
    st.subheader("In this Project I am Using the YOLO-NAS Object Detection Using Streamlit GUI App.")
    st.markdown("For More Details Of YOLO-NAS : https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md")
    st.markdown("This Application Developed by <a href='https://github.com/datamind321'>DataMind Platform 2.0</a>",unsafe_allow_html=True) 
		
		# st.markdown("If You have any queries , Contact Us On : ") 
    st.header("contact us on : bme19rahul.r@invertisuniversity.ac.in")

    st.divider()

  

    st.markdown("[![Linkedin](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/rahul-rathour-402408231/)")
		
    st.markdown("[![Instagram](https://img.icons8.com/color/1x/instagram-new.png)](https://instagram.com/_technical__mind?igshid=YmMyMTA2M2Y=)")


