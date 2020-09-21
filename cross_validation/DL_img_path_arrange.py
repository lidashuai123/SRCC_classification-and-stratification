#####在cohort文件中加入PVP图片的路径信息
import numpy as np
from sklearn.model_selection import KFold
import os
import pandas as pd


i = 1
path1 = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\fixed_rec_cropped_images.xlsx'
df1 = pd.read_excel(path1)
name1 = df1['ID'].tolist()
cohort1 = df1['PVP_save_path'].tolist()
path2 = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\cohort%s.csv'%str(i)
df2 = pd.read_csv(path2, encoding='gb18030')
name2 = df2['ID'].tolist()
cohort2 = []

for item in name2:
    cohort2.append(cohort1[name1.index(item)])

dict_info = {"ID": name2, "cohort": cohort2}

info = pd.DataFrame(dict_info)
file_name = 'DL_PVP_path%s.xlsx'%str(i)
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\DL_prob\DL_path'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)

info.to_excel(writer)
writer.save()
