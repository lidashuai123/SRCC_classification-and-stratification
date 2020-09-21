####临床变量显著性检验
import pandas as pd
import os
import tqdm
import numpy as np

source_file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
patients_df = pd.read_csv(source_file_path, encoding='gb18030')

patients_month = patients_df['month'].tolist()

patients_cohort = patients_df['cohort'].tolist()
patients_label = patients_df['label'].tolist()

patients_age = patients_df['age'].tolist()
patients_gender = patients_df['gender'].tolist()
patients_tumor_size = patients_df['size'].tolist()
patients_tumor_location = patients_df['tumor_location'].tolist()
patients_lauren = patients_df['lauren'].tolist()
patients_T = patients_df['T'].tolist()
patients_N = patients_df['N'].tolist()
patients_M = patients_df['M'].tolist()
patients_Chemo = patients_df['chemotherapy'].tolist()

label_train = []
label_test = []
age_train = []
age_test = []
gender_train = []
gender_test =[]
tumor_size_train = []
tumor_size_test = []
tumor_location_train =[]
tumor_location_test = []
lauren_train = []
lauren_test = []
T_train = []
T_test = []
N_train =[]
N_test = []
M_train = []
M_test = []
Chemo_train = []
Chemo_test = []

for i,la, a,g,ts,tl, l, t, n,m, c in zip(patients_cohort, patients_label, patients_age, patients_gender, patients_tumor_size, patients_tumor_location,
                              patients_lauren, patients_T, patients_N, patients_M, patients_Chemo):
    if la == 1:
        label_train.append(la)
        age_train.append(a)
        gender_train.append(g)
        tumor_size_train.append(ts)
        tumor_location_train.append(tl)
        lauren_train.append(l)
        T_train.append(t)
        N_train.append(n)
        M_train.append(m)
        Chemo_train.append(c)
    elif la == 0:
        label_test.append(la)
        age_test.append(a)
        gender_test.append(g)
        tumor_size_test.append(ts)
        tumor_location_test.append(tl)
        lauren_test.append(l)
        T_test.append(t)
        N_test.append(n)
        M_test.append(m)
        Chemo_test.append(c)


print('label_train')
print('1:', label_train.count(1), '0:', label_train.count(0))
print('label_test')
print('1:', label_test.count(1), '0:', label_test.count(0))


print('age_train')
print('median:', np.median(age_train), 'min:', min(age_train), 'max:', max(age_train))
print('age_test')
print('median:', np.median(age_test), 'min:', min(age_test), 'max:', max(age_test))

print('gender_train')
print('male:', gender_train.count(1), 'female:', gender_train.count(2))
print('gender_test')
print('male:', gender_test.count(1), 'female:', gender_test.count(2))

print('tumor_size_train')
print('median:', np.median(tumor_size_train), 'min:', min(tumor_size_train), 'max:', max(tumor_size_train))
print('tumor_size_test')
print('median:', np.median(tumor_size_test), 'min:', min(tumor_size_test), 'max:', max(tumor_size_test))

print('tumor_location_train')
print('AEG:', tumor_location_train.count(1)+tumor_location_train.count(4)+tumor_location_train.count(5),
      'Non_AEG:', tumor_location_train.count(2)+tumor_location_train.count(3)+tumor_location_train.count(6)+
      tumor_location_train.count(7)+tumor_location_train.count(8))
print('tumor_location_test')
print('AEG:', tumor_location_test.count(1)+tumor_location_test.count(4)+ tumor_location_test.count(5),
      'Non_AEG:', tumor_location_test.count(2)+tumor_location_test.count(3)+tumor_location_test.count(6)+
      tumor_location_test.count(7)+tumor_location_test.count(8))

print('Lauren_train')
print('In:', lauren_train.count(1), 'Diff:', lauren_train.count(2), 'Mix:', lauren_train.count(3), 'NA:', lauren_train.count(''))
print('Lauren_test')
print('In:', lauren_test.count(1), 'Diff:', lauren_test.count(2), 'Mix:', lauren_test.count(3), 'NA:', lauren_test.count(''))

print('T_train')
print('1:', T_train.count(1), '2:', T_train.count(2), '3:', T_train.count(3), '4:', T_train.count(4))
print('T_test')
print('1:', T_test.count(1), '2:', T_test.count(2), '3:', T_test.count(3), '4:', T_test.count(4))

print('N_train')
print('0:', N_train.count(0), '1:', N_train.count(1), '2:', N_train.count(2), '3:', N_train.count(3))
print('N_test')
print('0:', N_test.count(0), '1:', N_test.count(1), '2:', N_test.count(2), '3:', N_test.count(3))

print('M_train')
print('0:', M_train.count(0), '1:', M_train.count(1))
print('M_test')
print('0:', M_test.count(0), '1:', M_test.count(1))

print('Chemo_train')
print('1:', Chemo_train.count(1), '0:', Chemo_train.count(0))
print('Chemo_test')
print('1:', Chemo_test.count(1), '0:', Chemo_test.count(0))

print(len(age_train))
print(len(age_test))

print('ALL the size')
print('median:', np.median(patients_tumor_size), 'min:', min(patients_tumor_size), 'max:', max(patients_tumor_size))

print('month')
new_month = []
for i in patients_month:
    if i == i:
        new_month.append(i)
print('len:', len(new_month))
print('median:', np.median(new_month), 'min:', min(new_month), 'max:', max(new_month))