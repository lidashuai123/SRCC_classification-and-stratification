import numpy as np
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from utils.transforms import OneHotEncode


def getmax(img):
    hug = 0
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if img.getpixel((i, j)) > hug:
                hug = img.getpixel((i, j))
    return  hug



def load_image(file):
    return Image.open(file)


class Promise_data(Dataset):
    def __init__(self, file_path, img_transform = Compose([]),label_transform=Compose([]), co_transform=Compose([]),
                 labelled=True, valid=False):
        self.labelled = labelled
        self.valid = valid
        self.all_df = pd.read_excel(file_path)
        # print(self.all_df['valid'])
        if valid:##验证集
            self.df = self.all_df.loc[self.all_df['valid'] == True]###选择验证集
        else:###训练集，然后分为标签和非标签
            self.all_df = self.all_df.loc[self.all_df['valid'] == False]
            if labelled:
                self.df = self.all_df.loc[self.all_df['label'] == 'labelled']
        self.df = self.df.reset_index()###重新设置索引

        imgs = []
        for i in range(self.df.shape[0]):
            new_id = self.df.loc[i, 'ID']
            new_image_path = self.df.loc[i, 'PVP_save_path']
            new_seg_path = self.df.loc[i, 'PVP_seg_path']
            # new_image_path = self.df.loc[i, 'AP_save_path']
            # new_seg_path = self.df.loc[i, 'AP_seg_path']
            imgs.append((new_id, new_image_path, new_seg_path))

        self.imgs = imgs
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform

    def __getitem__(self, idx):
        new_id, new_image_path, new_seg_path = self.imgs[idx]
        img = load_image(new_image_path)
        label = load_image(new_seg_path)

        threshold = 0
        table = []
        for i in range(256):
            if i == threshold:
                table.append(0)
            else:
                table.append(1)
        label = label.point(table, '1')

        img, label = self.co_transform((img, label))

        img = self.img_transform(img)
        # print(f'getmax(label)::{getmax(label)}')
        label = self.label_transform(label)
        # print(label.max())
        # print(label.shape)
        ohlabel = OneHotEncode()(label)
        # if self.valid == True:
        #     return img, label, ohlabel, slice_id
        # elif self.valid == False and self.labelled == False:
        #     return img, label, ohlabel, slice_id

        return img, label, ohlabel, new_id

    def __len__(self):
        return len(self.imgs)
