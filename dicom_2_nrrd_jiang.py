import SimpleITK as sitk
import os
import xlrd
import pandas as pd


def conversion(in_file_path, out_path, name):
    # Dicom序列所在文件夹路径（在我们的实验中，该文件夹下有多个dcm序列，混合在一起）
    file_path = in_file_path

    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)

    # 查看该文件夹下的序列数量
    nb_series = len(series_IDs)
    print('病人序列的个数：%d' % nb_series)

    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[1]代表的是第二个序列的ID
    # 如果不添加series_IDs[1]这个参数，则默认获取第一个序列的所有切片路径
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[0])

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    # 查看该3D图像的尺寸
    print(image3D.GetSize())

    # 将序列保存为单个的DCM或者NRRD文件
    # sitk.WriteImage(image3D, 'img3D.dcm')
    newname = name + '.nrrd'
    # newname = name + '.nii'
    output = os.path.join(out_path, newname)
    sitk.WriteImage(image3D, output)

# # ###李敏的课题类型转换
# dir = r'F:\molecular_imaging\Limin\data\1CT400197'
# outpath = r'F:\molecular_imaging\Limin\data\1CT400197'
# patient = '1CT400197_ori'
# conversion(dir, outpath, patient)##dir是dicom文件存放的位置，output是输出的位置，patient存储文件的前缀名称

###蒋医生课题数据转换
base_path = r'F:\molecular_imaging\huaxi_jiang_yinjie\label'
base_out_path = r'I:\West China Hospotal-Gastric Cancer SRCC\NRRD'
series = 'pre' ##选择序列名称 AP pre　PVP
path = os.path.join(base_path, series, 'presample_nii_path.xlsx')##记得更改文件名称

outpath = os.path.join(base_out_path, series)

# # path = r'F:\molecular_imaging\huaxi_jiang\label_file\AP\sample_nii_all_AP.xlsx'
# # path = r'F:\molecular_imaging\huaxi_jiang\label_file\pre\sample_nii_all_pre.xlsx'
# path = r'F:\molecular_imaging\huaxi_jiang\label_file\PVP\sample_nii_all_PVP.xlsx'
# # outpath = r'H:\West China Hospital-Gastric Cancer\NRRD\AP'
# # outpath = r'H:\West China Hospital-Gastric Cancer\NRRD\pre'
# outpath = r'H:\West China Hospital-Gastric Cancer\NRRD\PVP'

data = pd.read_excel(path, sheet_name='Sheet1')
# data = pd.DataFrame(pd.read_excel(path, sheet_name='Sheet1'))
# name = list(data['name'])
# print(type(name))
name = data['name'].tolist()
name_uni = list(set(name))###保持顺序排除重复
name_uni.sort(key=name.index)
print(len(name_uni))



for patient in name_uni:
    number = patient.split('-')[0]
    print("病人编号 %s" % number)
    local_path = r'I:\West China Hospital-Gastric Cancer\R2 and DICOM'
    subdir = number + '-' + series##选择哪个序列的文件

    dir = os.path.join(local_path, patient, subdir)
    conversion(dir, outpath, patient)##dir是dicom文件存放的位置，output是输出的位置，patient存储文件的名称

