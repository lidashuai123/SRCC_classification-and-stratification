import pandas as pd
import os


path = r'F:\molecular_imaging\huaxi_jiang_yinjie\label\SRCC-WCH(2019-10-17)-修改.xlsx'
df = pd.read_excel(path, sheet_name='Sheet1')
print(type(df))

SRCC = df['SRCC'].tolist()
GENE = df['GENE'].tolist()
ID = df['ID'].tolist()
Number = df['Number'].tolist()

number_SRCC1 = 0##个数
number_SRCC2 = 0
number_SRCC3 = 0
number_GENE0 = 0
number_GENE1 = 0
number_GENEnan = 0
SRCC1_primary_path = []###存放原发灶分割文件的位置
SRCC2_primary_path = []
SRCC3_primary_path = []
GENE0_primary_path = []
GENE1_primary_path = []
GENENA_primary_path = []
primary_path = []
name = []
SRCC_label_new = []
GENE_label_new = []

local_path = r'I:\West China Hospital-Gastric Cancer\R2 and DICOM'
series = 'PVP' #定义序列名称 AP pre PVP
print(series)

# error_name = []
for SRCC_label, GENE_label, preid, postid in zip(SRCC, GENE, Number, ID):
    postid = str(int(postid)).zfill(10)
    preid = str(preid)
    dir_name = preid + '-' + postid##病人级别的文件夹名称
    sub_dir_name = preid + '-' + series##得到序列级别的文件夹名称

    sub_dir = os.listdir(local_path)
    if dir_name not in sub_dir:
        print(dir_name)
        print('not exit subdir')
        # error_name.append(dir_name)
        os._exit(0)##如果病人不存在，则终止程序
# dict_error_name = pd.DataFrame({"name": error_name})
# file_to_save = pd.ExcelWriter(r"F:\molecular_imaging\huaxi_jiang_yinjie\error_name.xlsx")
# dict_error_name.to_excel(file_to_save)
# file_to_save.save()

    dir = os.path.join(local_path, dir_name, sub_dir_name)
    primary_name = sub_dir_name + '_Merge.nii'
    if primary_name in os.listdir(dir):
        primary = os.path.join(dir, primary_name)
        if SRCC_label == 1:
            number_SRCC1 += 1
            SRCC1_primary_path.append(primary)
        elif SRCC_label == 2:
            number_SRCC2 += 1
            SRCC2_primary_path.append(primary)
        elif SRCC_label == 3:
            number_SRCC3 += 1
            SRCC3_primary_path.append(primary)

        if GENE_label == 0:
            number_GENE0 += 1
            GENE0_primary_path.append(primary_path)
        elif GENE_label == 1:
            number_GENE1 += 1
            GENE1_primary_path.append(primary_path)
        elif str(GENE_label) == 'nan':
            GENE_label = 'NAN'
            number_GENEnan += 1
            GENENA_primary_path.append(primary_path)
        primary_path.append(primary)
        name.append(dir_name)
        SRCC_label_new.append(SRCC_label)
        GENE_label_new.append(GENE_label)
    else:
        print(primary_name)

print(f"number_SRCC1: {number_SRCC1}, number_SRCC2: {number_SRCC2}, number_SRCC3: {number_SRCC3}")
print(f"number_GENE0: {number_GENE0}, number_GENE1: {number_GENE1}")

dict_info = {"name": name, "SRCC_label": SRCC_label_new, "GENE_label": GENE_label_new, "path": primary_path}

info = pd.DataFrame(dict_info)
file_name = series + 'sample_nii_path.xlsx'
file_save_path = r'F:\molecular_imaging\huaxi_jiang_yinjie\label'
file_save_path = os.path.join(file_save_path, series, file_name)
writer = pd.ExcelWriter(file_save_path)

info.to_excel(writer)
writer.save()