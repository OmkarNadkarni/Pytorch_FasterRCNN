import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

def DisplayResult(model,device,image_path,iou_threshold=0.7,conf_threshold=0.5):
    img = Image.open(img_path)
    trans = transforms.Compose([transforms.ToTensor()])
    torch_img = trans(img)

    #print(torch_img)
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([torch_img.to(device)])
    #print(prediction)
    im = transforms.ToPILImage()(torch_img).convert("RGB")
    prediction2 = torchvision.ops.nms(prediction[0]['boxes'],scores=prediction[0]['scores'],iou_threshold=iou_threshold)
    indices = prediction2.tolist()
    bboxes = prediction[0]['boxes'].tolist()
    labels = prediction[0]['labels'].tolist()
    scores = prediction[0]['scores'].tolist()
    newimg = cv2.imread(img_path)

    for index in indices:
        if scores[index]>conf_threshold :
            box = bboxes[index]
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax=int(box[3])

            if labels[index]==1:
                label_text = 'bolted'
                cv2.putText(newimg,label_text,(xmin,ymin-2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                cv2.rectangle(newimg,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            else:
                label_text='un_bolted'
                cv2.putText(newimg,label_text,(xmin,ymin-2),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
                cv2.rectangle(newimg,(xmin,ymin),(xmax,ymax),(255,0,0),2)

    return newimg


CHECKPOINT_PATH = r'/home/omkarnadkarni/FlangeDetection/MODEL_CHECKPOINT/model_best20.pt'
DIR = r'/home/omkarnadkarni/FlangeDetection/data/inf/'
#img_path = r'/home/omkarnadkarni/FlangeDetection/data/test/images/un_bolted43.jpeg'
imgs = [DIR+i for i in os.listdir(DIR)]
iou_threshold = 0.7
conf_threshold = 0.8

#define the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

checkpoint = torch.load(CHECKPOINT_PATH,map_location=device)
model = checkpoint['model']
model.load_state_dict(checkpoint['state_dict'])
img_list = []
#loading Image for prediction
for img_path in imgs:
    img_list.append(DisplayResult(model,device,img_path,iou_threshold,conf_threshold))


f, axarr = plt.subplots(2,2,figsize=(15,15))
axarr[0,0].imshow(img_list[0])
axarr[0,1].imshow(img_list[1])

axarr[1,0].imshow(img_list[2])
axarr[1,1].imshow(img_list[3])
plt.show()
