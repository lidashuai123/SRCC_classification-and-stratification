from torchcam import open_image, image2batch, int2tensor, getCAM
from torchvision.models import resnet18, resnet50
import torch
from torchvision import models, transforms
import os
import torchvision.transforms as T
from my_resnet import my_resnet18
import tqdm
import pandas as pd
import cv2
import shutil
import numpy as np


file_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\fixed_rec_cropped_images_activate.xlsx'

df = pd.read_excel(file_path)
df = df.loc[df['label'] == 0]
filenames = df['PVP_save_path'].tolist()
label = df['label'].tolist()

for image_path, temp_label in zip(filenames, label):

    # if 'mask' in image_path or 'grad' in image_path:
    #     continue
    # ID = image_path.split('.')[0]
    # print(ID)
    # image_paths = os.path.join(image_paths_folder, image_path)

    ID = os.path.basename(image_path).split('.')[0]
    img = open_image(image_path, (112, 112))
    temp_label = str(temp_label)
    # img = valid_transforms(img)
    input = image2batch(img)
    image_class = 0  # cat class in imagenet
    # model = my_resnet50()

    model = my_resnet18()
    model.load_state_dict(
        torch.load(r'E:\pycharm_project\Huaxi_Yinjie_Jiang\AP_weights\epoch_9_712_704.pth')
    )

    target = int2tensor(image_class)
    gray_img, color_img = getCAM(model, img, input, target, ID=ID, display=False, save=True)

    gray_path = 'E:\\Radiomics\\huaxi_jiang_yinjie\\cam\\DL_cam\\SRCC\\%s_gray_%s.png' % (ID, temp_label)
    color_path = 'E:\\Radiomics\\huaxi_jiang_yinjie\\cam\\DL_cam\\SRCC\\%s_cam_%s.png' % (ID, temp_label)
    ori_path = 'E:\\Radiomics\\huaxi_jiang_yinjie\\cam\\DL_cam\\SRCC\\%s_%s.png' % (ID, temp_label)
    img2 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    # print(img2.shape)
    img2 = cv2.resize(img2, (112, 112))
    # print(img2.shape)
    cv2.imwrite(ori_path, img2)
    cv2.imwrite(gray_path, gray_img)
    cv2.imwrite(color_path, color_img)
    # del gray_img, color_img, input, model, target
    # torch.cuda.empty_cache()

