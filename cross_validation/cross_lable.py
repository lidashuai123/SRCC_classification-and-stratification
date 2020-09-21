###生成交叉验证的train 和test标记

import numpy as np
from sklearn.model_selection import KFold
import os
import pandas as pd

path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\for_crossv_alidation.csv'
df = pd.read_csv(path, encoding='gb18030')

name = df['name'].tolist()
id = df['id'].tolist()
name_txt = df['name_txt'].tolist()
SRCC_label = df['SRCC_label'].tolist()
label = df['label'].tolist()

kf = KFold(n_splits=5, shuffle=True)
j = 0
for train, test in kf.split(name):
    j += 1
    cohort = []
    for i in range(len(name)):
        if i in train:
            cohort.append('train')
        else:
            cohort.append('test')
    # print(cohort)
    # print("%s %s" % (train, test))
    # print(type(train))

    dict_info = {"name": name, "id": id, "name_txt": name_txt, "SRCC_label": SRCC_label, "label": label, "cohort": cohort}

    info = pd.DataFrame(dict_info)
    file_name = 'cohort%s.xlsx'%str(j)
    print(file_name)
    file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts'
    file_save_path = os.path.join(file_save_path, file_name)
    writer = pd.ExcelWriter(file_save_path)

    info.to_excel(writer)
    writer.save()