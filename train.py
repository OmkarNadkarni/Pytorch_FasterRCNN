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

TRAIN_DIR = r'data/train'
csv_path = TRAIN_DIR+'/train.csv'
TEST_DIR = r'data/test'
test_csv = TEST_DIR+'/test.csv'

#creating Flange Dataset class

class FlangeDataset(torch.utils.data.Dataset):
    def __init__(self,root,csv_path,transforms=None):
        self.root =root
        self.transforms = transforms
        self.df = pd.read_csv(csv_path)
        self.imgs = list(sorted(os.listdir(os.path.join(root,"images"))))

    def __getitem__(self,idx):
        img_path = os.path.join(self.root,"images",self.imgs[idx])
        PILimg = Image.open(img_path).convert("RGB")
        img = img_path.split('/')[-1]
        rows = self.df.loc[self.df['filename']==img]
        num_objs = len(rows)
        bbox = []
        label_list = []
        target = {}
        if num_objs>0:
            #get bounding box
            xmin = rows['xmin'].values
            xmax = rows['xmax'].values
            ymin = rows['ymin'].values
            ymax = rows['ymax'].values
            labels = rows['class'].values
            target["iscrowd"] = torch.ones((num_objs,), dtype=torch.int64)
            for i in range(len(xmin)):
                bbox.append([xmin[i],ymin[i],xmax[i],ymax[i]])
                if labels[i]=='bolted':
                    label_list.append(1)
                else:
                    label_list.append(2)
        else:
            bbox.append([0,0,0.1,0.1])
            label_list.append(0)
            target["iscrowd"] = torch.ones((1,), dtype=torch.int64)
            #convert to tensors
        boxes = torch.as_tensor(bbox,dtype=torch.float32)
        labels = torch.as_tensor(label_list,dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #torch.as_tensor(height*width,dtype=torch.float32)


        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area


        if self.transforms is not None:
            PILimg,target = self.transforms(PILimg,target)
        return PILimg,target
    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


#Function to display images with bounding boxes in a grid **will only display 20 images

def DisplayResults(model,dataset_test):


    image_datas = []
    for img,_ in dataset_test:
        #img, _ = dataset_test[4]
        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])

        im = transforms.ToPILImage()(img).convert("RGB")
        prediction2 = torchvision.ops.nms(prediction[0]['boxes'],scores=prediction[0]['scores'],iou_threshold=0.3)
        indices = prediction2.tolist()
        bboxes = prediction[0]['boxes'].tolist()
        labels = prediction[0]['labels'].tolist()
        scores = prediction[0]['scores'].tolist()
        threshold = 0.5
        open_cv_image = np.array(im)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        for index in indices:
            if scores[index]>threshold :
                box = bboxes[index]
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax=int(box[3])
                cv2.rectangle(open_cv_image,(xmin,ymin),(xmax,ymax),(255,0,0),2)
                if labels[index]==1:
                    label_text = '1'
                else:
                    label_text='2'
                cv2.putText(open_cv_image,label_text,(xmin,ymin-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
        image_datas.append(open_cv_image)
        #plt.imshow(open_cv_image)

    while True:
        if len(image_datas)<20:
            image_datas.append(image_datas[0])
        else:
            break


    f, axarr = plt.subplots(5,4,figsize=(15,15))
    axarr[0,0].imshow(image_datas[0])
    axarr[0,1].imshow(image_datas[1])
    axarr[0,2].imshow(image_datas[2])
    axarr[0,3].imshow(image_datas[3])

    axarr[1,0].imshow(image_datas[4])
    axarr[1,1].imshow(image_datas[5])
    axarr[1,2].imshow(image_datas[6])
    axarr[1,3].imshow(image_datas[7])

    axarr[2,0].imshow(image_datas[8])
    axarr[2,1].imshow(image_datas[9])
    axarr[2,2].imshow(image_datas[10])
    axarr[2,3].imshow(image_datas[11])

    axarr[3,0].imshow(image_datas[12])
    axarr[3,1].imshow(image_datas[13])
    axarr[3,2].imshow(image_datas[14])
    axarr[3,3].imshow(image_datas[15])

    axarr[4,0].imshow(image_datas[16])
    axarr[4,1].imshow(image_datas[17])
    axarr[4,2].imshow(image_datas[18])
    axarr[4,3].imshow(image_datas[19])

    #training faster RCNN model for training

dataset = FlangeDataset(TRAIN_DIR,csv_path,get_transform(train=True))
dataset_test = FlangeDataset(TEST_DIR,test_csv,get_transform(train=False))

EPOCHS = 20
LEARNING_RATE = 0.001


# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
test_indices = torch.randperm(len(dataset_test)).tolist()
# evens = list(range(0, len(dataset), 2))
# odds = list(range(1, len(dataset), 2))
dataset = torch.utils.data.Subset(dataset, indices)
dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2,collate_fn=utils.collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Whether using GPU or CPU? ",device)

num_classes = 3  #2 classes +1 background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

model.to(device)
#construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
for epoch in range(EPOCHS):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

#save the model that is trained for inference
PATH = r'model'+str(EPOCHS)+'.pt'
torch.save(model,PATH)
