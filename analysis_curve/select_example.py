####印戒细胞癌课题，将PVP的最大层面移动到一个文件夹中方便选出合适的肿瘤
import os
import shutil
import numpy as np

path = r'G:\West China Hospotal-Gastric Cancer SRCC\Max_ROI_save\PVP'
target_path = r'G:\West China Hospotal-Gastric Cancer SRCC\Max_ROI_save\select_example'
for folder in os.listdir(path):
    for item in os.listdir(os.path.join(path, folder)):
        shutil.copy(os.path.join(path, folder, item), target_path)
