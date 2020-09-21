import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
df = pd.read_csv(file_path, encoding='gb18030')


cohort = df['cohort'].tolist()
label = df['label'].tolist()
label_txt = df['label_txt'].tolist()

R = df['PVP_prob'].tolist()
DL = df['DL_prob'].tolist()
Unet = df['Unet_prob'].tolist()
Clinical = df['clinical_prob'].tolist()
Merge = df['R_DL_C_merge'].tolist()

out = R + DL + Unet + Clinical + Merge
category = label_txt + label_txt + label_txt + label_txt + label_txt

a = ['R model'] * len(R)
b = ['DL model'] * len(R)
c = ['Unet encoder model'] * len(R)
d = ['Clinical model'] * len(R)
e = ['Merge model'] * len(R)

model = a + b + c + d + e

dict_info = {"Probability": out, "Category": category, "Model types": model}
new_info = pd.DataFrame(dict_info)

sns.set(font_scale=1.2)
sns.set_style('white')
# sns.swarmplot(x="model", y="out", hue="category", data=new_info)
# sns.boxplot(x = 'model', y= 'out', data = new_info, hue = 'category')

# fig = sns.catplot(x="Model", y="Probability", hue="Category",
#             kind="box", aspect=1.5, dodge=True, data=new_info)

fig = sns.catplot(x="Model types", y="Probability", hue="Category",
            kind="violin", split=True, palette="pastel", aspect=1.7, data=new_info)
leg = fig._legend
# leg.set_bbox_to_anchor([0.55, 0.9])
# sns.catplot(x="model", y="out", hue="category", aspect=2, kind="swarm", data=new_info)


# sns.catplot(x="model", y="out", hue="category",
#             kind="violin", split=True,
#             palette="pastel", data=new_info)


# plt.show()
save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph'
filename = 'model_distribution.tiff'
plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
