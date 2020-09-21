####整理R模型的概率值

import numpy as np
from sklearn.model_selection import KFold
import os
import pandas as pd


i = 4
path1 = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\Rmodel\R_outcome%s.csv'%str(i)
df1 = pd.read_csv(path1, encoding='gb18030')
name1 = df1['name'].tolist()
prob1 = df1['prob'].tolist()

path2 = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\cohort%s.csv'%str(i)
df2 = pd.read_csv(path2, encoding='gb18030')
name2 = df2['ID'].tolist()
prob2 = []

for item in name2:
    prob2.append(prob1[name1.index(item)])

dict_info = {"name": name2, "R_prob": prob2}

info = pd.DataFrame(dict_info)
file_name = 'R_arrange_outcome%s.xlsx'%str(i)
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\R_prob'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)

info.to_excel(writer)
writer.save()
