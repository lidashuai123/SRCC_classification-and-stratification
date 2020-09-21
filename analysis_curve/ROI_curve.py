###此文件用于勾画病灶的区域
import cv2
import os
import numpy as np


# img_path = r'E:\Radiomics\huaxi_jiang_yinjie\cam\raw_images\101-0003657545\101-0003657545.png'
# mask_path = r'E:\Radiomics\huaxi_jiang_yinjie\cam\raw_images\101-0003657545\101-0003657545_mask.png'
# img_path =r'E:\Radiomics\huaxi_jiang_yinjie\cam\raw_images\297-0006269941\297-0006269941.png'
# mask_path =r'E:\Radiomics\huaxi_jiang_yinjie\cam\raw_images\297-0006269941\297-0006269941_mask.png'

# E:\Radiomics\huaxi_jiang_yinjie\cam\selected\101-0003657545_0.png
# E:\Radiomics\huaxi_jiang_yinjie\cam\selected\101-0003657545.png
#
# E:\Radiomics\huaxi_jiang_yinjie\cam\selected\297-0006269941_0.png
# E:\Radiomics\huaxi_jiang_yinjie\cam\selected\297-0006269941.png

# img_path =r'E:\Radiomics\huaxi_jiang_yinjie\cam\selected\101-0003657545_0.png'
# mask_path =r'E:\Radiomics\huaxi_jiang_yinjie\cam\selected\101-0003657545.png'

# img_path =r'E:\Radiomics\huaxi_jiang_yinjie\cam\selected\297-0006269941_0.png'
# mask_path =r'E:\Radiomics\huaxi_jiang_yinjie\cam\selected\297-0006269941.png'


img_path =r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph\ROI_curve\345-0006769690.png'
mask_path =r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph\ROI_curve\345-0006769690_mask.png'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = np.array(mask,np.uint8)

# cv2.imshow('img', img)  #显示原始图像
# cv2.waitKey()

print(mask.shape)

ret, thresh = cv2.threshold(mask, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# img = cv2.drawContours(img, contours, -1, (0,0,255), 3)
###只保留ROI曲线
image = np.zeros([400,400,3],np.uint8)
img = cv2.drawContours(image, contours, -1, (0,0,255), 3)

# cv2.imshow('img', img)  #显示原始图像
# cv2.waitKey()
filepath = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph\ROI_curve\ROI2.png'
cv2.imwrite(filepath, img)