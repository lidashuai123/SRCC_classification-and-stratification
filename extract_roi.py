'''
此文件致力于将病灶roi提取出来，分为四个方案：
外接矩形，抠出病灶（固定矩形大小），抠出病灶（外接矩形），固定矩形大小
'''
import os
import cv2
import pandas as pd


def REC_generate(mask, image):
    x, y, w, h = cv2.boundingRect(mask)
    # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 2)###图像上画框
    # cv2.imshow('rec', mask)
    # cv2.waitKey(0)
    cropped_image = image[y:y+h, x:x+w] # 裁剪坐标为[y0:y1, x0:x1]
    # cv2.imshow('crop', cropped_image)
    # cv2.waitKey(0)
    return cropped_image


def pure_rec_generate(mask, image):
    mask = mask // 255
    dot_image = mask * image
    cropped_image = REC_generate(mask, dot_image)
    return cropped_image



def fixed_rec_generate(mask, image):
    fixed_length = 112
    x, y, w, h = cv2.boundingRect(mask)
    center_x = x + w//2
    center_y = y + h//2
    cropped_image = image[center_y - fixed_length//2:center_y + fixed_length//2, center_x - fixed_length//2:center_x + fixed_length//2]  # 裁剪坐标为[y0:y1, x0:x1]
    # cv2.imshow('crop', cropped_image)
    # cv2.waitKey(0)
    return cropped_image


def pure_fixed_generate(mask, image):##先点乘后裁剪
    mask = mask//255
    dot_image = mask * image
    cropped_image = fixed_rec_generate(mask, dot_image)
    return cropped_image

##图片路径
base_local_path = r'G:\West China Hospotal-Gastric Cancer SRCC\Max_ROI_save'
series = 'PVP' ### 更改序列AP pre PVP，记得更改文件的名称
##读取文件统计label
base_label_file = r'E:\Radiomics\huaxi_jiang_yinjie\label'
label_file = os.path.join(base_label_file, series, 'PVPsample_nii_path.xlsx')###更改文件目录
label_info = pd.DataFrame(pd.read_excel(label_file, index_col='name'))

base_save_path = r'G:\West China Hospotal-Gastric Cancer SRCC\cropped_images'
local_path = os.path.join(base_local_path, series)

excel_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file'
excel_save_path = os.path.join(excel_save_path, series)

crop_type_list = ['rec', 'fixed_rec', 'pure_fixed', 'pure_rec']##四种模型：rec:自己的外接矩形  fixed_rec：固定112大小的外接矩形 pure_fixed: 单纯的病灶固定窗口大小 pure_rec :单纯病灶外接外接矩阵

for crop_type in crop_type_list:
    # crop_type = 'pure_fixed'
    ###裁剪图片保存路径
    save_path = os.path.join(base_save_path, series, crop_type)

    patient_id = []
    patient_SRCC_label = []
    patient_GENE_label = []
    image_save_path = []
    for ID_folder in os.listdir(local_path):
        ID_path = os.path.join(local_path, ID_folder)

        patient_id.append(ID_folder)
        patient_SRCC_label.append(label_info.loc[ID_folder, 'SRCC_label'])
        patient_GENE_label.append(label_info.loc[ID_folder, 'GENE_label'])

        flag = 0
        for item in os.listdir(ID_path):
            suffix = item.split('.')[-1]
            if suffix == 'png' and 'mask' in item:
                mask_path = os.path.join(ID_path, item)
                flag += 1
            elif suffix == 'png' and 'mask' not in item:
                image_path = os.path.join(ID_path, item)
                flag += 1
            else:
                continue

        if flag == 2: ##判断是否同时含有mask和image
            mask = cv2.imread(mask_path, 0)##参数0是读取灰度图，得到的是ndarray

            print(f'{crop_type},{mask.shape}')
            image = cv2.imread(image_path, 0)
        else:
            print(ID_folder)
            os._exit(0)


        if crop_type == 'rec':
            cropped_image = REC_generate(mask, image)
        elif crop_type == 'fixed_rec':
            cropped_image = fixed_rec_generate(mask, image)
        elif crop_type == 'pure_fixed':
            cropped_image = pure_fixed_generate(mask, image)
        elif crop_type == 'pure_rec':
            cropped_image = pure_rec_generate(mask, image)

        file_name = ID_folder + '.png'
        file_save_path = os.path.join(save_path, ID_folder)
        image_save_path.append(os.path.join(file_save_path, file_name))
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)

        # cv2.imwrite(os.path.join(file_save_path, file_name), cropped_image)

    dict_info = {'ID': patient_id, "SRCC_label": patient_SRCC_label, 'GENE_label': patient_GENE_label, 'save_path': image_save_path}
    dataframe_info = pd.DataFrame(dict_info)
    excel_name = series + '_' + crop_type + '_' + 'cropped_images.xlsx'
    writer = pd.ExcelWriter(os.path.join(excel_save_path, excel_name))
    dataframe_info.to_excel(writer)
    writer.save()




