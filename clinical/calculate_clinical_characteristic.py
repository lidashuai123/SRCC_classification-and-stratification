#####此文件用于计算基础的临床变量统计学信息 印戒细胞癌课题
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import os


file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
df = pd.read_csv(file_path, encoding='gb18030')

cohort = df['cohort'].tolist()
label = df['label'].tolist()
R = df['PVP_prob'].tolist()
DL = df['DL_prob'].tolist()
Unet = df['Unet_prob'].tolist()
Clinical = df['clinical_prob'].tolist()
Merge = df['R_DL_C_merge'].tolist()

age = df['age'].tolist()
gender = df['gender'].tolist()
location = df['location'].tolist()

men_age = []
women_age = []
for item, i in zip(gender, age):
    if item == 1:
        men_age.append(i)
    elif item == 2:
        women_age.append(i)

train_age = []
valid_age = []
for item, i in zip(cohort, age):
    if item == 'train':
        train_age.append(i)
    elif item == 'test':
        valid_age.append(i)