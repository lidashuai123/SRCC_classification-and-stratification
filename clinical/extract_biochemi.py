###印戒细胞癌课题生化指标提取
import pandas as pd
import os
import tqdm

target_file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
source_file_path = r'E:\Radiomics\huaixi_jiang_LNM\clinical_files\血指标\select_clinical_file_operation.xlsx'

target_df = pd.read_csv(target_file_path, encoding='gb18030')
source_df = pd.read_excel(source_file_path, sheet_name='Sheet1')

variables = ['ID', '癌胚抗原', '甲胎蛋白', '血清CA19-9', '血清CA15-3', '血清CA-125', '糖类抗原72-4', '白蛋白', '总蛋白',
             'AST/ALT', '肌酐', '尿素']

final_df = pd.DataFrame(columns=variables)
delta_df = pd.DataFrame(columns=variables)

for i in tqdm.tqdm(range(target_df.shape[0])):
    ID = target_df.loc[i, 'id'].split('-')[-1]
    ID = int(ID)
    temp_df = source_df.loc[source_df['PADMNO登记号'] == ID]
    temp_df = temp_df.reset_index()  ##重新设置索引

    if temp_df.shape[0] == 0:
        print(ID)

    # for j in range(delta_df.shape[1]):
    #     delta_df.iloc[0, j] = 'NAN'
    delta_df.loc[0] = 'NAN'###增减一行
    delta_df.iloc[0, 0] = ID

    for k in range(temp_df.shape[0]):
        if temp_df.loc[k, '化验结果项目名称'] in variables:
            index = variables.index(temp_df.loc[k, '化验结果项目名称'])
            delta_df.iloc[0, index] = temp_df.loc[k, '定量结果']
    # print(delta_df)
    # final_df = pd.concat(final_df, delta_df)
    final_df = final_df.append(delta_df, ignore_index=True)

print(final_df.shape[0])


new_info = pd.DataFrame(final_df)
file_name = 'biochemi.xlsx'
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob'
if not os.path.exists(os.path.join(file_save_path)):
    os.makedirs(os.path.join(file_save_path))
file_save_path = os.path.join(file_save_path, file_name)

writer = pd.ExcelWriter(file_save_path)

new_info.to_excel(writer)
writer.save()