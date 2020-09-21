###检查印戒细胞癌数据的ID是否包含在生化信息中的ID
import pandas as pd
import os
import tqdm

target_file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
source_file_path = r'E:\Radiomics\huaixi_jiang_LNM\clinical_files\血指标\select_clinical_file_operation.xlsx'


target_df = pd.read_csv(target_file_path, encoding='gb18030')
source_df = pd.read_excel(source_file_path, sheet_name='Sheet1')

target_ID = target_df['id'].tolist()
source_ID = source_df['PADMNO登记号'].tolist()
source_ID = [str(x) for x in source_ID]

for item in target_ID:
    suffix = item.split('-')[-1]
    if suffix not in source_ID:
        print(item)
