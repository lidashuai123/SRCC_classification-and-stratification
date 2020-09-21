####此文件用于将原来的路径文件顺序按照R语言的随机顺序排列分为训练集和验证集

import pandas as pd
import numpy as np
import os


target_file_path  = r'E:\Radiomics\huaxi_jiang_yinjie\R_code\auc_test\arranged_order\chongzu.csv'
source_file_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\fixed_rec_cropped_images.xlsx'

target_df = pd.read_csv(target_file_path, encoding='gb18030')
source_df = pd.read_excel(source_file_path)

source_cohort = []
source_name = source_df['ID'].tolist()

target_cohort = target_df['cohort'].tolist()
target_name = target_df['name'].tolist()

for item in source_name:
    index = target_name.index(item)
    source_cohort.append(target_cohort[index])


dict_info = {"name": source_name, "cohort": source_cohort}

new_info = pd.DataFrame(dict_info)
file_name = 'fixed_rec_cropped_images_arranged.xlsx'
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)
new_info.to_excel(writer)
writer.save()
