from super_gradients.training import models
import torch 


# --------------------------- Select Device ----------------------------

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


# --------------------------- Load Model ----------------------------

model=models.get('yolo_nas_s',pretrained_weights='coco').to(device)

# ---------------------------- Image Detection ----------------------------------------------

out=model.predict('images/pexels-nic-law-792831.jpg')


# --------------------------- Video Detection ----------------------------

# out=model.predict('videos/city.mp4').save('videos/output_video/')

#--------------------------- live webcam ---------------------------------

# model.predict_webcam()



