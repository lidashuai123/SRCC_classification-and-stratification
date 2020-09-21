'''
此文件用于计算图片的均值和方差
'''
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm


def main(opt_dir):
    opt = opt_dir
    img_filenames = []
    for i in opt:
        sub_list = os.listdir(i)
        new_list = [os.path.join(i, j) for j in sub_list]
        img_filenames = img_filenames + new_list
    # img_filenames = os.listdir(opt.dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint8), -1)
        # img = cv2.imread(opt.dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((1,)))###图片的通道数
        s_list.append(s.reshape((1,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])

if __name__ == '__main__':
    path = r'G:\West China Hospotal-Gastric Cancer SRCC\cropped_images\PVP'
    opt_dir = []
    for type_folder in os.listdir(path):
        print(type_folder)
        folder_path = os.path.join(path, type_folder)

        for item in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, item)):
                opt_dir.append(os.path.join(folder_path, item))
        main(opt_dir)

