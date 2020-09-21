import pandas as pd
import os
import shutil


file_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\pure_fixed_cropped_images.xlsx'
df = pd.read_excel(file_path)
ID = df['ID'].tolist()
path = df['PVP_save_path'].tolist()


file_path2 = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\fixed_rec_cropped_images_activate.xlsx'
df2 = pd.read_excel(file_path2)
# df2 = df2.loc[df2['label'] == 1]
ID2 = df2['ID'].tolist()

save_path = r'E:\Radiomics\huaxi_jiang_yinjie\cam\DL_cam\SRCC'
for item in ID2:
    index = ID.index(item)
    img_path = path[index]
    shutil.copy(img_path, save_path)
