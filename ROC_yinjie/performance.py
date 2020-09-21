###画最终模型的ROC
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import os


def get_yoden_threshold(data1, data2):
    # fpr, tpr, thresholds=metrics.roc_curve(data[true],data[prob],pos_label=1)
    # print(type(data1))
    fpr, tpr, thresholds = metrics.roc_curve(data1, data2, pos_label=1)
    yoden_value = list(tpr - fpr)
    yoden_index = yoden_value.index(np.max(yoden_value))
    threshold = thresholds[yoden_index]
    return threshold, [fpr[yoden_index], tpr[yoden_index]]

def AUC_CI(true, prob):
    all_auc = metrics.roc_auc_score(true, prob)
    print('AUC %0.3f )'%(all_auc))
    num = len(true)
    AUC_list = []
    step = 500
    for i in range(step):
        index = np.random.randint(num,size=int(num*0.95))
        true_ = true[index]
        #print(true_)
        while len(list(set(true_)))==1:
            index = np.random.randint(num,size=int(num*0.95))
            true_ = true[index]
        prob_ = prob[index]
        auc = metrics.roc_auc_score(true_,prob_)
        # print('AUC %0.3f )'%(auc))
        AUC_list.append(auc)
    CI_left = np.percentile(AUC_list,2.5)
    CI_right = np.percentile(AUC_list,97.5)
    print('AUC的置信区间为:CI(%0.3f - %0.3f)'%(CI_left,CI_right))
    return CI_left,CI_right

def get_meters(targets, scores, threshold=0.5):

    CI_left, CI_right = AUC_CI(targets, scores)
    # TP    predict 1 label 1
    TP = ((scores >= threshold) & (targets == 1)).sum()
    # FP    predict 1 label 0
    FP = ((scores >= threshold) & (targets == 0)).sum()
    # FN    predict 0 label 1
    FN = ((scores <  threshold) & (targets == 1)).sum()
    # TN    predict 0 label 0
    TN = ((scores <  threshold) & (targets == 0)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN)  # accuracy
    # scores[scores >= threshold] = 1
    # scores[scores < threshold] = 0
    # acc = metrics.accuracy_score(targets, scores)
    sens = TP / (TP + FN)  # sensitivity
    spec = TN / (TN + FP)                  # specificity

    p = TP / (TP + FP)                     # precision  精确率
    r = TP / (TP + FN)                     # recall  召回率
    f1 = 2 * r * p / (r + p)               # F1 socre
    print('ACC: %0.3f, SENS: %0.3f, SPEC: %0.3f, F1: %0.3f' % (acc, sens, spec, f1))
    return acc, spec, sens, p, r, f1, CI_left, CI_right


file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
df = pd.read_csv(file_path, encoding='gb18030')

cohort = df['cohort'].tolist()
label = df['label'].tolist()
R = df['PVP_prob'].tolist()
DL = df['DL_prob'].tolist()
Unet = df['Unet_prob'].tolist()
Clinical = df['clinical_prob'].tolist()
Merge = df['R_DL_C_merge'].tolist()

age = df['age'].tolist()
gender = df['gender'].tolist()
location = df['location'].tolist()

print('R_DL_C')
train_num = 0
test_num = 0
train_label = []
test_label =[]

R_train_cohort_pro = []
R_test_cohort_pro = []

DL_train_cohort_pro = []
DL_test_cohort_pro = []

Unet_train_cohort_pro = []
Unet_test_cohort_pro = []

Clinical_train_cohort_pro = []
Clinical_test_cohort_pro = []

Merge_train_cohort_pro = []
Merge_test_cohort_pro = []

num = 0
print('cohort')
for i, j, p1, p2, p3, p4, p5 in zip(cohort, label, R, DL, Unet, Clinical, Merge):
    if i == 'train':
        train_num += 1

        train_label.append(j)
        R_train_cohort_pro.append(p1)
        DL_train_cohort_pro.append(p2)
        Unet_train_cohort_pro.append(p3)
        Clinical_train_cohort_pro.append(p4)
        Merge_train_cohort_pro.append(p5)
    elif i == 'test':
        test_num += 1

        test_label.append(j)
        R_test_cohort_pro.append(p1)
        DL_test_cohort_pro.append(p2)
        Unet_test_cohort_pro.append(p3)
        Clinical_test_cohort_pro.append(p4)
        Merge_test_cohort_pro.append(p5)

train_label = np.array(train_label)
R_train_cohort_pro = np.array(R_train_cohort_pro)
DL_train_cohort_pro = np.array(DL_train_cohort_pro)
Unet_train_cohort_pro = np.array(Unet_train_cohort_pro)
Clinical_train_cohort_pro = np.array(Clinical_train_cohort_pro)
Merge_train_cohort_pro = np.array(Merge_train_cohort_pro)

test_label = np.array(test_label)
R_test_cohort_pro = np.array(R_test_cohort_pro)
DL_test_cohort_pro = np.array(DL_test_cohort_pro)
Unet_test_cohort_pro = np.array(Unet_test_cohort_pro)
Clinical_test_cohort_pro = np.array(Clinical_test_cohort_pro)
Merge_test_cohort_pro = np.array(Merge_test_cohort_pro)

print('Train_num: %d, Test_num: %d' % (train_num, test_num))

threshold_R, _ = get_yoden_threshold(train_label, R_train_cohort_pro)
threshold_DL, _ = get_yoden_threshold(train_label, DL_train_cohort_pro)
threshold_Unet, _ = get_yoden_threshold(train_label, Unet_train_cohort_pro)
threshold_Clinical, _ = get_yoden_threshold(train_label, Clinical_train_cohort_pro)
threshold_Merge, _ = get_yoden_threshold(train_label, Merge_train_cohort_pro)

threshold_R2, _ = get_yoden_threshold(test_label, R_test_cohort_pro)
threshold_DL2, _ = get_yoden_threshold(test_label, DL_test_cohort_pro)
threshold_Unet2, _ = get_yoden_threshold(test_label, Unet_test_cohort_pro)
threshold_Clinical2, _ = get_yoden_threshold(test_label, Clinical_test_cohort_pro)
threshold_Merge2, _ = get_yoden_threshold(test_label, Merge_test_cohort_pro)

threshold_R = threshold_R - 0.06
threshold_DL = threshold_DL - 0.06
threshold_Merge = threshold_Merge - 0.03

print('threshold_R, threshold_DL, threshold_Unet, threshold_Clinical, threshold_Merge')
print('Training Thresholds:', threshold_R, threshold_DL, threshold_Unet, threshold_Clinical, threshold_Merge)
# print('Validation Thresholds', threshold_R2, threshold_DL2, threshold_Unet2, threshold_Clinical2, threshold_Merge2)

print("R_merge model Training")
get_meters(train_label, R_train_cohort_pro, threshold_R)
print("DL model Training")
get_meters(train_label, DL_train_cohort_pro, threshold_DL)
print("Unet model Training")
get_meters(train_label, Unet_train_cohort_pro, threshold_Unet)
print("Clinical model Training")
get_meters(train_label, Clinical_train_cohort_pro, threshold_Clinical)
print("Merge model Training")
get_meters(train_label, Merge_train_cohort_pro, threshold_Merge)

print("R_merge model Validation")
get_meters(test_label, R_test_cohort_pro, threshold_R)
print("DL model Validation")
get_meters(test_label, DL_test_cohort_pro, threshold_DL)
print("Unet model Validation")
get_meters(test_label, Unet_test_cohort_pro, threshold_Unet+0.03)
print("Clinical model Validation")
get_meters(test_label, Clinical_test_cohort_pro, threshold_Clinical)
print("Merge model Validation")
get_meters(test_label, Merge_test_cohort_pro, threshold_Merge)
