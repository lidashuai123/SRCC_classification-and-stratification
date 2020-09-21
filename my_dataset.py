'''
此文件中来定义自己的dataset,印戒细胞癌
'''

import torch as t
from torch.utils import data
import pandas as pd
from PIL import Image
import cv2
import torchvision.transforms as T


class Yinjie_data(data.Dataset):
    def __init__(self, file_path, transforms=None, train=True, test=True):
        self.train = train
        self.test = test
        self.df = pd.read_excel(file_path)
        # self.df = self.df.sample(frac=1, random_state=12)##打乱顺序,并设置种子点
        # self.df = self.df.reset_index()  ##重新设置索引

        if self.train and not self.test:
            self.df = self.df.loc[self.df['cohort'] == 'train']
            print('train')
            print(self.df.shape)
        elif not self.train and self.test:
            self.df = self.df.loc[self.df['cohort'] == 'test']
            print('test')
            print(self.df.shape)
        self.df = self.df.reset_index()  ##重新设置索引

        imgs = []
        for i in range(self.df.shape[0]):
            ID = self.df.loc[i, 'ID']
            SRCC_label = self.df.loc[i, 'SRCC_label']
            # GENE_label = self.df.loc[i, 'GENE_label']
            # AP_path = self.df.loc[i, 'AP_save_path']
            # pre_path = self.df.loc[i, 'pre_save_path']
            PVP_path = self.df.loc[i, 'pre_save_path']
            imgs.append((ID, SRCC_label, PVP_path))

        self.imgs = imgs
        self.transform = transforms
        # img_len = self.df.shape[0]
        # train_ratio = 0.7
        # valid_ratio = 0.8##累加比例

    def __getitem__(self, idx):
        ID, SRCC_label, PVP_path = self.imgs[idx]
        SRCC_label = int(SRCC_label)
        label = 0 if SRCC_label == 1 or SRCC_label == 2 else 1
        # label = 0 if SRCC_label == 1 else 1
        # print(label)

        # AP_img = Image.open(AP_path)
        # pre_img = Image.open(pre_path)
        # PVP_img = Image.open(PVP_path)
        # img = Image.merge('RGB', (AP_img, pre_img, PVP_img))

        
        img = Image.open(PVP_path)###PVP
        # img = Image.open(AP_path)  ###AP

        # img2 = cv2.imread(AP_path)
        # print(img2.max())
        # img3 = T.ToTensor()(img2)
        # print(img3.max())
        if self.transform is not None:
            img = self.transform(img)
            # print(img.max())
            # print(img.size())
        return img, label, ID

    def __len__(self):
        return len(self.imgs)



