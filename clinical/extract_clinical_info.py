####提取出患者的年龄，性别和肿瘤的位置
import pandas as pd
import os
import numpy as np


target_file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
source_file_path = r'E:\Radiomics\huaixi_jiang_LNM\clinical_files\Clinical file operation.xlsx'

target_df = pd.read_csv(target_file_path, encoding='gb18030')
source_df = pd.read_excel(source_file_path, sheet_name='file')

source_id = source_df['住院号'].tolist()
print(source_id)
source_id = [str(x) for x in source_id]
print(source_id)
source_age = source_df['age'].tolist()
source_gender = source_df['gender'].tolist()
source_location = source_df['location'].tolist()
source_TNM = source_df['TNM'].tolist()
source_size = source_df['size'].tolist()
source_tumor_location = source_df['tumor_location'].tolist()
source_lauren = source_df['lauren'].tolist()
source_T = source_df['T'].tolist()
source_N = source_df['N'].tolist()
source_M = source_df['M'].tolist()


target_id = target_df['id'].tolist()
print(target_id)
age = []
gender = []
location = []
TNM = []
size = []
tumor_location = []
lauren = []
T = []
N = []
M = []

for item in target_id:
    suffix = item.split('-')[-1]
    print(suffix)
    index = source_id.index(suffix)
    age.append(source_age[index])
    gender.append(source_gender[index])
    location.append(source_location[index])
    TNM.append(source_TNM[index])
    size.append(source_size[index])
    tumor_location.append(source_tumor_location[index])
    lauren.append(source_lauren[index])
    T.append(source_T[index])
    N.append(source_N[index])
    M.append(source_M[index])


dict_info = {'id': target_id, 'age': age, 'gender': gender, 'location': location, 'TNM': TNM, 'size': size, 'tumor_location': tumor_location,
             'lauren': lauren, 'T': T, 'N': N, 'M': M}
info = pd.DataFrame(dict_info)
file_name = 'all_clinical.xlsx'
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)
info.to_excel(writer)
writer.save()
