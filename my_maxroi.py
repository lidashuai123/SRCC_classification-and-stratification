
'''
导出最大层面并统计mask的长宽
导出图像分为medicl和natural两种，可以更改
df和series需要根据序列名称进行更改
enable_calculation 参数确定是否输出病灶尺寸的信息
level ,window 控制窗位和窗宽
'''

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy.misc as misc
from PIL import Image


def boundingBox( A, use2D=False):
    B = np.argwhere(A)
    if use2D == True:
        (ystart, xstart), (ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
        return (ystart, xstart), (ystop, xstop)
    else:
        (zstart, ystart, xstart), (zstop, ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
        return (zstart, ystart, xstart), (zstop, ystop, xstop)


def window_level(image_array):
    level = 40 ##设置窗位，相当于HU值的均值
    window = 260 #设置窗宽，相当于HU值的范围
    min_HU = level - window/2
    max_HU = level + window/2
    dfactor = 255.0/(max_HU - min_HU)
    # np.clip(image_array, min_HU, max_HU, image_array)
    for i in np.arange(image_array.shape[0]):
        for j in np.arange(image_array.shape[1]):
            image_array[i, j] = int((image_array[i, j] - min_HU) * dfactor)
    min_index = image_array < 0
    image_array[min_index] = 0
    max_index = image_array > 255
    image_array[max_index] = 255
    return image_array

def binary_255(image_array):
    index = image_array == 1
    image_array[index] = 255
    return image_array

def deep_features(imageFilepath,maskFilepath, saveEachSlicePath, name, image_type='medical'):
    imageFile = imageFilepath
    print(f'处理的第{num}个病人为:',os.path.basename(imageFilepath))

    nrrd = sitk.ReadImage(imageFile)
    nrrd_array3d = sitk.GetArrayFromImage(nrrd) # z, y, x
    nii = sitk.ReadImage(maskFilepath)
    nii_array = sitk.GetArrayFromImage(nii) # z, y, x

    dims,rows,cols=nii_array.shape #a.shape返回的元组表示该数组的维数、行数与列数
    print(f'shape:{dims, rows, cols}')

    (zstart, ystart, xstart), (zstop, ystop, xstop) = boundingBox(nii_array, use2D=False)

    # for i in range(zstart,zstop):###部分医院存在图片翻转的情况
    #     nii_array[i] = nii_array[i][::-1]   ## for guangdong the nii need Flip vertical Along y axis

    ct = nii_array
    max_area_idx = sorted([[idx, ct[idx].sum()] for idx in range(ct.shape[0])], key=lambda x: x[1], reverse=True)##根据CT面积大小排序，从大到小
    print(max_area_idx[0])
    index = max_area_idx[0][0]##得到最大面对应的层数
    pixel_sum = max_area_idx[0][1]
    print(f'最大层的层数为:{index}')

    nrrd_array = nrrd_array3d[index]
    nrrd_array = window_level(nrrd_array)###设置窗位和窗宽
    nii_array = nii_array[index]
    nii_array = binary_255(nii_array)##二值图片像素值更改

    if image_type == 'medical':
        dcmsliceOut = sitk.GetImageFromArray(nrrd_array)
        niisliceOut = sitk.GetImageFromArray(nii_array)
        nrrdsavepath = f'{saveEachSlicePath}/{name}/{name}.nrrd'
        niisavepath = f'{saveEachSlicePath}/{name}/{name}.nii'
        if not os.path.exists(os.path.dirname(nrrdsavepath)):
            os.makedirs(os.path.dirname(nrrdsavepath))  # create each subfold in path
        sitk.WriteImage(dcmsliceOut, nrrdsavepath)
        sitk.WriteImage(niisliceOut, niisavepath)
    elif image_type == 'natural':
        dcmsliceOut = Image.fromarray(nrrd_array)
        dcmsliceOut = dcmsliceOut.convert('L')
        niisliceOut = Image.fromarray(nii_array)
        niisliceOut = niisliceOut.convert('1') ##保存为二值图像
        # dcmsliceOut = misc.toimage(nrrd_array)
        # niisliceOut = misc.toimage(nii_array)
        nrrdsavepath = f'{saveEachSlicePath}/{name}/{name}.png'
        niisavepath = f'{saveEachSlicePath}/{name}/{name}_mask.png'
        if not os.path.exists(os.path.dirname(nrrdsavepath)):
            os.makedirs(os.path.dirname(nrrdsavepath))  # create each subfold in path
        dcmsliceOut.save(nrrdsavepath)
        niisliceOut.save(niisavepath)

    return zstart, ystart, xstart,zstop, ystop, xstop, pixel_sum


if __name__ == '__main__':
    local_path = r'G:\West China Hospotal-Gastric Cancer SRCC'
    base_allPatientsPath = os.path.join(local_path, 'NRRD')  ##原始数据

    df = pd.read_excel(r'G:\West China Hospotal-Gastric Cancer SRCC\Max_ROI_save\presample_nii_path.xlsx')##记得更改存储分割路径的文件'AP pre PVP'
    allROIPath = df['path'].tolist()##ROI文件路径， 列表形式
    base_saveEachSlicePath = os.path.join(local_path, 'Max_ROI_save')  ##存储路径
    series = 'pre'  ##选择序列‘AP’pre PVP
    image_type = 'natural'###选择图片保存类型  medical natural
    enable_calculation = False ###选择是否计算和输出病灶尺寸的统计信息


    allPatientsPath = os.path.join(base_allPatientsPath, series)
    saveEachSlicePath = os.path.join(base_saveEachSlicePath, series)

    num = 1
    x_Range = []
    y_Range = []
    xy_Multi = []
    Pixel_sum = []
    for Image_name in os.listdir(allPatientsPath):
        name = str(Image_name.split('.')[0])
        preid = str(name.split('-')[0])
        if '.nrrd' in Image_name:
            Image_Path = os.path.join(allPatientsPath, Image_name)
        else:
            print(f'No such .nrrd file: {Image_name}')

        roi_name = preid + '-' + series
        for item in allROIPath:
            if roi_name in item:
                Roi_Path = item
                break
            else:
                continue

        zstart, ystart, xstart, zstop, ystop, xstop, pixel_sum = deep_features(Image_Path, Roi_Path, saveEachSlicePath, name, image_type)  ##传入图片和分割文件的路径
        if enable_calculation:
            x_range = abs(xstop - xstart)
            y_range = abs(ystop - ystart)
            xy_multi = x_range * y_range
            x_Range.append(x_range)
            y_Range.append(y_range)
            xy_Multi.append(xy_multi)
            Pixel_sum.append(pixel_sum)
        num += 1

    if enable_calculation:
        dict = {'x_Range': x_Range, 'y_Range': y_Range, 'xy_Multi': xy_Multi, 'Pixel_sum': Pixel_sum}
        info = pd.DataFrame(dict)
        file_name = series + 'size_mask.xlsx'
        file_save_path = r'F:\molecular_imaging\huaxi_jiang_yinjie\统计信息'
        file_save_path = os.path.join(file_save_path, file_name)
        writer = pd.ExcelWriter(file_save_path)
        info.to_excel(writer)
        writer.save()

