import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd


INPUT_DIR = r'/home/omkarnadkarni/od_pytorch/data/ub_images/'
OUTPUT_DIR = r'/home/omkarnadkarni/od_pytorch/data/images/'

input_imgs_path = [INPUT_DIR+i for i in os.listdir(INPUT_DIR)]
index= 0
for img_path in input_imgs_path:
    index+=1
    img = Image.open(img_path)
    rgb_img = img.convert('RGB')
    image_name = 'un_bolted'+str(index)+'.jpeg'
    SAVE_PATH = OUTPUT_DIR+image_name
    rgb_img.save(SAVE_PATH)
