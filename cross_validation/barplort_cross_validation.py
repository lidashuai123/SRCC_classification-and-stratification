####绘制交叉验证的结果分布

# import seaborn as sns
# sns.set(style="ticks", palette="pastel")
#
# # Load the example tips dataset
# tips = sns.load_dataset("tips")
#
# # Draw a nested boxplot to show bills by day and time
# sns.boxplot(x="day", y="total_bill",
#             hue="smoker", palette=["m", "g"],
#             data=tips)
# sns.despine(offset=10, trim=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\zhengli.xlsx'
df = pd.read_excel(file_path)

new_info = df

sns.set(font_scale=1.2)
sns.set_style('white')
sns.set(style="ticks", palette="pastel")

# sns.swarmplot(x="model", y="out", hue="category", data=new_info)
# sns.boxplot(x = 'model', y= 'out', data = new_info, hue = 'category')

# fig = sns.catplot(x="Model", y="Probability", hue="Category",
#             kind="box", aspect=1.5, dodge=True, data=new_info)

# sns.catplot(x="model", y="auc", hue="category",
#             kind="bar",  palette=["m", "g"], aspect=1.4, data=new_info)#split=True,

sns.barplot(x="Model", y="AUC", hue="Category",
            palette=["m", "g"], data=new_info)
sns.swarmplot(x="Model", y="AUC", hue="Category",
              palette=["r", "b"], data=new_info)




# sns.catplot(x="model", y="auc", hue="category",
#             kind="swarm",  palette=["b", "b"], aspect=1.4, data=new_info)#split=True,
# leg = fig._legend
plt.ylim(0.55, 0.85)
sns.despine(offset=0.5, trim=True)
# leg.set_bbox_to_anchor([0.55, 0.9])
# sns.catplot(x="model", y="out", hue="category", aspect=2, kind="swarm", data=new_info)


# sns.catplot(x="model", y="out", hue="category",
#             kind="violin", split=True,
#             palette="pastel", data=new_info)


# plt.show()
save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts'
filename = 'cross_validation.pdf'
plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
