"""
提取一致的生存期数据
"""
import pandas as pd
import os

target_file = r'E:\Radiomics\huaxi_jiang_yinjie\feature\PVP\radiomics_SRCC_PVP_Primary_operation.csv'
source_file = r'E:\Radiomics\huaxi_jiang_yinjie\label\12.25更新随访时间\1225_operation.xlsx'

target_df = pd.read_csv(target_file)
source_df = pd.read_excel(source_file)

name = target_df['name'].tolist()
target_id = [x.split('-')[0] for x in name]

id = []
name = []
month = []
status = []
SRCC = []
Gene = []
chemotherapy = []
for index, row in source_df.iterrows():
    if str(row['NO']) in target_id:
        id.append(str(row['NO']) + '-' + str(row['id']))
        name.append(row['name'])
        month.append(row['month'])
        status.append(row['status'])
        SRCC.append(row['SRCC'])
        Gene.append(row['Gene'])
        chemotherapy.append(row['chemotherapy'])

dict_info = {'id': id, 'name': name, 'month': month, 'status':status, 'SRCC': SRCC, 'Gene': Gene, 'chemotherapy': chemotherapy}
info = pd.DataFrame(dict_info)
file_name = 'select_surv_info2.xlsx'
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\exe'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)
info.to_excel(writer)
writer.save()