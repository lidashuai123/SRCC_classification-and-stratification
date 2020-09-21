####此文件用于建立Rmodel时与深度学习模型的cohort完成对应
import numpy as np
from sklearn.model_selection import KFold
import os
import pandas as pd


i = 5
path1 = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\cohort%s.xlsx'%str(i)
df1 = pd.read_excel(path1)
name1 = df1['name'].tolist()
cohort1 = df1['cohort'].tolist()
path2 = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\Rmodel\radiomics_SRCC_PVP_Primary_operation%s.csv'%str(i)
df2 = pd.read_csv(path2, encoding='gb18030')
name2 = df2['name'].tolist()
cohort2 = []

for item in name2:
    cohort2.append(cohort1[name1.index(item)])

dict_info = {"name": name2, "cohort": cohort2}

info = pd.DataFrame(dict_info)
file_name = 'R_cohort%s.xlsx'%str(i)
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\Rmodel'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)

info.to_excel(writer)
writer.save()
