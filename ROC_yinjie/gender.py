###画最终模型的ROC
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import os
plt.rc('font', family='Times New Roman')

def AUC_CI(true, prob):
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
    spec = TN / (TN + FP)                  # specificity
    sens = TP / (TP + FN)                  # sensitivity
    p = TP / (TP + FP)                     # precision  精确率
    r = TP / (TP + FN)                     # recall  召回率
    f1 = 2 * r * p / (r + p)               # F1 socre

    return acc, spec, sens, p, r, f1, CI_left, CI_right


def get_roc_and_meters(values, curves_name, save_path, filename, threshold=0.5):
    '''
    Parameters:
    values: the shape is [num_curves, 2, num_samples]. Two values (targets, predicted_scores) per sample per curve (group).
    curves_name: curves' name.
    threshold: drived by Youden index from the training set, used to be the cut-off of positive and negetive samples.
    '''

    # fig initialization
    fig = plt.figure(figsize=(7.5, 6), dpi=300)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.plot([0, 1], [0, 1], color=[0.6, 0.6, 0.6], lw=0.8, linestyle='--')
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # get roc, auc, and other meters
    for idx, (targets, scores) in enumerate(values):
        fpr, tpr, _ = metrics.roc_curve(targets, scores)
        auc = metrics.roc_auc_score(targets, scores, average=None)
        acc, spec, sens, p, r, f1, CI_left, CI_right = get_meters(targets, scores, threshold)

        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)
        plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.3f; %0.3f-%0.3f)' % (curves_name[idx], auc, CI_left, CI_right))
        plt.title('Receiver operating characteristic curves', fontsize=17)
        # plt.plot(fpr, tpr, 'r-', lw=3, label='%s (AUC = %0.3f; %0.3f-%0.3f)' % (curves_name, auc, CI_left, CI_right))
        # plt.plot(fpr, tpr, 'r-', lw=3, label='training cohort (AUC = %0.3f; %0.3f-%0.3f)' % (auc, CI_left, CI_right))
        print('AUC: %0.3f' % auc)
        # get other meters
        # acc, spec, sens, p, r, f1
        print('ACC: %0.3f, SPEC: %0.3f, SENS: %0.3f, F1: %0.3f' % (acc, spec, sens, f1))
        # print(get_meters(targets, scores, threshold))
    # R1_sens = [0.869]
    # R1_spec = [1-0.450]
    # plt.scatter(R1_spec, R1_sens, color='k', marker='p', s=30, alpha=0.2)

    plt.legend(loc=4, fontsize=12)

    # plt.savefig(os.path.join(save_path, file_name))
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.show()

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
test_label = []

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
print('gender')
for i, j, p1 in zip(gender, label, Merge):
    if i == 1:
        train_num += 1
        train_label.append(j)
        Merge_train_cohort_pro.append(p1)
    elif i == 2:
        test_num += 1
        test_label.append(j)
        Merge_test_cohort_pro.append(p1)

train_label = np.array(train_label)
Merge_train_cohort_pro = np.array(Merge_train_cohort_pro)

test_label = np.array(test_label)
Merge_test_cohort_pro = np.array(Merge_test_cohort_pro)

print('Male: %d, Female: %d' % (train_num, test_num))
input_values = [[train_label, Merge_train_cohort_pro], [test_label, Merge_test_cohort_pro]]

input_names = ['Male', 'Female']

save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph'
if not os.path.exists(save_path):
    os.makedirs(save_path)
file_name = 'gender.tiff'
get_roc_and_meters(input_values, input_names, save_path, file_name, threshold=0.278)

dict_info_train = {'train_label': train_label, 'Merge_train_cohort_pro': Merge_train_cohort_pro}
dict_info_test = {'test_label': test_label, 'Merge_test_cohort_pro': Merge_test_cohort_pro}
info_path_train = pd.DataFrame(dict_info_train)
info_path_test = pd.DataFrame(dict_info_test)

file_path_save_path_train = os.path.join(r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph\stratified_sheet', 'gender_train.xlsx')
writer_train = pd.ExcelWriter(file_path_save_path_train)
info_path_train.to_excel(writer_train)
writer_train.save()

file_path_save_path_test = os.path.join(r'E:\Radiomics\huaxi_jiang_yinjie\outcome\graph\stratified_sheet', 'gender_test.xlsx')
writer_test = pd.ExcelWriter(file_path_save_path_test)
info_path_test.to_excel(writer_test)
writer_test.save()
