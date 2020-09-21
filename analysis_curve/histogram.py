import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# 将图片读取为数组
img_path =r'E:\Radiomics\huaixi_jiang_LNM\outcome\ROI_curve\histgram.png'
im = np.array(Image.open(img_path))
# 创建画板
plt.figure("Lena")
# 将图片数组扁平化，成为一维数组，供hist方法使用
arr = im.flatten()
# # 绘制直方图
# n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='blue', alpha=1)
# # plt(kind = 'kde', color = 'red', label = '核密度图')


arr = np.array(arr)
sns.distplot(arr, kde_kws={"color": "r", "lw": 3},
             hist_kws={"linewidth": 3,
                       "alpha": 1, "color": "b"})
plt.show()
# 展示直方图
plt.show()


