###此文件用于统计交叉验证的性能结果
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
    return CI_left,CI_right, all_auc

def get_meters(targets, scores, threshold=0.5):

    CI_left, CI_right, all_auc = AUC_CI(targets, scores)
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
    return all_auc, acc, spec, sens, p, r, f1, CI_left, CI_right



Rauc = []
Racc = []
Rsens = []
Rspec = []
Rleft = []
Rright = []
Dauc = []
Dacc = []
Dsens = []
Dspec = []
Dleft = []
Dright = []
Uauc = []
Uacc = []
Usens = []
Uspec = []
Uleft = []
Uright = []
Cauc = []
Cacc = []
Csens = []
Cspec = []
Cleft = []
Cright = []
Mauc = []
Macc = []
Msens = []
Mspec = []
Mleft = []
Mright = []
Rauc_t = []
Racc_t = []
Rsens_t = []
Rspec_t = []
Rleft_t = []
Rright_t = []
Dauc_t = []
Dacc_t = []
Dsens_t = []
Dspec_t = []
Dleft_t = []
Dright_t = []
Uauc_t = []
Uacc_t = []
Usens_t = []
Uspec_t = []
Uleft_t = []
Uright_t = []
Cauc_t = []
Cacc_t = []
Csens_t = []
Cspec_t = []
Cleft_t = []
Cright_t = []
Mauc_t = []
Macc_t = []
Msens_t = []
Mspec_t = []
Mleft_t = []
Mright_t = []
for i in range(1, 6):

    # file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob\proba.csv'
    file_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\cohort%s.csv'%str(i)
    df = pd.read_csv(file_path, encoding='gb18030')

    cohort = df['cohort'].tolist()
    label = df['label'].tolist()
    R = df['R_prob'].tolist()
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

    threshold_R = threshold_R - 0.08
    # threshold_DL = threshold_DL - 0.06
    # threshold_Merge = threshold_Merge - 0.03

    print('threshold_R, threshold_DL, threshold_Unet, threshold_Clinical, threshold_Merge')
    print('Training Thresholds:', threshold_R, threshold_DL, threshold_Unet, threshold_Clinical, threshold_Merge)
    # print('Validation Thresholds', threshold_R2, threshold_DL2, threshold_Unet2, threshold_Clinical2, threshold_Merge2)


    print("R_merge model Training")
    all_auc1, acc1, spec1, sens1, p1, r1, f11, CI_left1, CI_right1 = get_meters(train_label, R_train_cohort_pro, threshold_R)
    Rauc.append(all_auc1)
    Racc.append(acc1)
    Rsens.append(sens1)
    Rspec.append(spec1)
    Rleft.append(CI_left1)
    Rright.append(CI_right1)



    print("DL model Training")
    all_auc2, acc2, spec2, sens2, p2, r2, f12, CI_left2, CI_right2 = get_meters(train_label, DL_train_cohort_pro, threshold_DL)
    Dauc.append(all_auc2)
    Dacc.append(acc2)
    Dsens.append(sens2)
    Dspec.append(spec2)
    Dleft.append(CI_left2)
    Dright.append(CI_right2)


    print("Unet model Training")
    all_auc3, acc3, spec3, sens3, p3, r3, f13, CI_left3, CI_right3 = get_meters(train_label, Unet_train_cohort_pro, threshold_Unet)
    Uauc.append(all_auc3)
    Uacc.append(acc3)
    Usens.append(sens3)
    Uspec.append(spec3)
    Uleft.append(CI_left3)
    Uright.append(CI_right3)



    print("Clinical model Training")
    all_auc4, acc4, spec4, sens4, p4, r4, f14, CI_left4, CI_right4 =  get_meters(train_label, Clinical_train_cohort_pro, threshold_Clinical)
    Cauc.append(all_auc4)
    Cacc.append(acc4)
    Csens.append(sens4)
    Cspec.append(spec4)
    Cleft.append(CI_left4)
    Cright.append(CI_right4)



    print("Merge model Training")
    all_auc5, acc5, spec5, sens5, p5, r5, f15, CI_left5, CI_right5 = get_meters(train_label, Merge_train_cohort_pro, threshold_Merge)
    Mauc.append(all_auc5)
    Macc.append(acc5)
    Msens.append(sens5)
    Mspec.append(spec5)
    Mleft.append(CI_left5)
    Mright.append(CI_right5)








    print("R_merge model Validation")
    all_auc6, acc6, spec6, sens6, p6, r6, f16, CI_left6, CI_right6 = get_meters(test_label, R_test_cohort_pro, threshold_R)
    Rauc_t.append(all_auc6)
    Racc_t.append(acc6)
    Rsens_t.append(sens6)
    Rspec_t.append(spec6)
    Rleft_t.append(CI_left6)
    Rright_t.append(CI_right6)


    print("DL model Validation")
    all_auc7, acc7, spec7, sens7, p7, r7, f17, CI_left7, CI_right7 = get_meters(test_label, DL_test_cohort_pro, threshold_DL)
    Dauc_t.append(all_auc7)
    Dacc_t.append(acc7)
    Dsens_t.append(sens7)
    Dspec_t.append(spec7)
    Dleft_t.append(CI_left7)
    Dright_t.append(CI_right7)




    print("Unet model Validation")
    all_auc8, acc8, spec8, sens8, p8, r8, f18, CI_left8, CI_right8 = get_meters(test_label, Unet_test_cohort_pro, threshold_Unet+0.03)
    Uauc_t.append(all_auc8)
    Uacc_t.append(acc8)
    Usens_t.append(sens8)
    Uspec_t.append(spec8)
    Uleft_t.append(CI_left8)
    Uright_t.append(CI_right8)



    print("Clinical model Validation")
    all_auc9, acc9, spec9, sens9, p9, r9, f19, CI_left9, CI_right9 = get_meters(test_label, Clinical_test_cohort_pro, threshold_Clinical)
    Cauc_t.append(all_auc9)
    Cacc_t.append(acc9)
    Csens_t.append(sens9)
    Cspec_t.append(spec9)
    Cleft_t.append(CI_left9)
    Cright_t.append(CI_right9)


    print("Merge model Validation")
    all_auc10, acc10, spec10, sens10, p10, r10, f110, CI_left10, CI_right10 = get_meters(test_label, Merge_test_cohort_pro, threshold_Merge)
    Mauc_t.append(all_auc10)
    Macc_t.append(acc10)
    Msens_t.append(sens10)
    Mspec_t.append(spec10)
    Mleft_t.append(CI_left10)
    Mright_t.append(CI_right10)



dict_info = {"Rauc": Rauc, "Racc": Racc, "Rsens": Rsens, "Rspec": Rspec, "Rleft": Rleft, "Rright": Rright}
info = pd.DataFrame(dict_info)
file_name = 'R_train1.xlsx'
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)
info.to_excel(writer)
writer.save()
dict_info = {"Rauc_t": Rauc_t, "Racc_t": Racc_t, "Rsens_t": Rsens_t, "Rspec_t": Rspec_t, "Rleft_t": Rleft_t, "Rright_t": Rright_t}
info = pd.DataFrame(dict_info)
file_name = 'R_test1.xlsx'
file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
file_save_path = os.path.join(file_save_path, file_name)
writer = pd.ExcelWriter(file_save_path)
info.to_excel(writer)
writer.save()
#
#
# dict_info = {"Dauc": Dauc, "Dacc": Dacc, "Dsens": Dsens, "Dspec": Dspec, "Dleft": Dleft, "Dright": Dright}
# info = pd.DataFrame(dict_info)
# file_name = 'D_train.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
# dict_info = {"Dauc_t": Dauc_t, "Dacc_t": Dacc_t, "Dsens_t": Dsens_t, "Dspec_t": Dspec_t, "Dleft_t": Dleft_t, "Dright_t": Dright_t}
# info = pd.DataFrame(dict_info)
# file_name = 'D_test.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
#
#
# dict_info = {"Uauc": Uauc, "Uacc": Uacc, "Usens": Usens, "Uspec": Uspec, "Uleft": Uleft, "Uright": Uright}
# info = pd.DataFrame(dict_info)
# file_name = 'U_train.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
# dict_info = {"Uauc_t": Uauc_t, "Uacc_t": Uacc_t, "Usens_t": Usens_t, "Uspec_t": Uspec_t, "Uleft_t": Uleft_t, "Uright_t": Uright_t}
# info = pd.DataFrame(dict_info)
# file_name = 'U_test.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
#
#
# dict_info = {"Cauc": Cauc, "Cacc": Cacc, "Csens": Csens, "Cspec": Cspec, "Cleft": Cleft, "Cright": Cright}
# info = pd.DataFrame(dict_info)
# file_name = 'C_train.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
# dict_info = {"Cauc_t": Cauc_t, "Cacc_t": Cacc_t, "Csens_t": Csens_t, "Cspec_t": Cspec_t, "Cleft_t": Cleft_t, "Cright_t": Cright_t}
# info = pd.DataFrame(dict_info)
# file_name = 'C_test.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
#
#
# dict_info = {"Mauc": Mauc, "Macc": Macc, "Msens": Msens, "Mspec": Mspec, "Mleft": Mleft, "Mright": Mright}
# info = pd.DataFrame(dict_info)
# file_name = 'M_train.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()
# dict_info = {"Mauc_t": Mauc_t, "Macc_t": Macc_t, "Msens_t": Msens_t, "Mspec_t": Mspec_t, "Mleft_t": Mleft_t, "Mright_t": Mright_t}
# info = pd.DataFrame(dict_info)
# file_name = 'M_test.xlsx'
# file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\cross_validation\cohorts\performance'
# file_save_path = os.path.join(file_save_path, file_name)
# writer = pd.ExcelWriter(file_save_path)
# info.to_excel(writer)
# writer.save()