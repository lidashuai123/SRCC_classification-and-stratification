"""
此文件用于进行验证模型测试
"""
from my_resnet import my_resnet18 as my_model
import torch
import torch.nn as nn
import torchvision.transforms as T
from my_dataset import Yinjie_data
from torch.utils.data.sampler import WeightedRandomSampler
import os
from torchvision import models
from torch import optim
from torch.optim.lr_scheduler import *
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
# from sklearn.metrics import roc_auc_score
from visdom import Visdom
import pandas as pd
import datetime


def get_yoden_threshold(data1, data2):
    # fpr, tpr, thresholds=metrics.roc_curve(data[true],data[prob],pos_label=1)
    # print(type(data1))
    fpr, tpr, thresholds = roc_curve(data1, data2, pos_label=1)
    yoden_value = list(tpr - fpr)
    yoden_index = yoden_value.index(np.max(yoden_value))
    threshold = thresholds[yoden_index]
    return threshold, [fpr[yoden_index], tpr[yoden_index]]

def cal_acc(thred, out,label):
    out[out > thred] = 1
    out[out <= thred] = 0
    acc = accuracy_score(label, out)
    TP = sum((out == label) & (label == 1))
    TN = sum((out == label) & (label == 0))
    FP = sum((out != label) & (label == 0))
    FN = sum((out != label) & (label == 1))
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    # print('Sensitivity: %.03f | Specifity: %.03f '% (TP / (TP + FN), TN.double() / (TN + FP)))
    return acc, Sen, Spe

def my_softmax(data):
    # print(data)
    for i in range(data.shape[0]):
        data[i] = np.exp(data[i])/sum(np.exp(data[i]))
        # print(data[i])
    # print(data)
    return data


train_transforms = T.Compose([
    T.Resize((112, 112)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.RandomRotation(45),
    T.ToTensor(),
    # T.Normalize(mean=[0.3],
    #             std=[0.2])
    T.Normalize(mean=[0.459],
                std=[0.250])
])

valid_transforms = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    # T.Normalize(mean=[0.3],
    #             std=[0.2])
    T.Normalize(mean=[0.459],
                std=[0.250])
])


use_gpu = torch.cuda.is_available()
print(use_gpu)

file_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\fixed_rec_cropped_images.xlsx'

test_dataset = Yinjie_data(
    file_path, transforms=valid_transforms, train=False,
    test=False)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
##加载自己更改的resnet结构,加载预训练权重
model = my_model()

pretrained_weight_path = r'E:\pycharm_project\Huaxi_Yinjie_Jiang\AP_weights\epoch_9_712_704.pth'

pretrained_dict = torch.load(pretrained_weight_path)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
if len(pretrained_dict) != 0:
    print('加载成功')
# pretrained_dict = torch.load(pretrained_weight_path)
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# if len(pretrained_dict) != 0:
#     print('加载成功')

model = model.cuda()

criterion = nn.CrossEntropyLoss()
lr = 1e-6##原始是1e-3
weight_decay = 1e-2##原始的是1
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = StepLR(optimizer, gamma=0.5, step_size=15)##先用固定学习率 gamma是学习率衰减的倍数
criterion.cuda()
epoch = 100


def my_test(epoch):
    print("\nTest Epoch: %d" % epoch)
    model.eval()
    with torch.no_grad():
        all_label = np.array([])
        all_out = np.array([])
        all_name = np.array([])
        for batch_idx, (img, label, ID) in enumerate(testloader):
            image = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(image)
            ##计算auc
            ##使用Crossentropy损失时
            out_array = out.data.cpu().numpy()
            out_prob = my_softmax(out_array)

            out_auc = out_prob[:, 1]  ##取出其中的一列用于计算auc
            all_out = np.hstack([all_out, out_auc])
            label_array = label.cpu().numpy()
            all_label = np.hstack([all_label, label_array])
            all_name = np.hstack([all_name, ID])

            loss = criterion(out, label)
            # viz.line([loss.item()], [global_step * len(testloader)+ batch_idx], win='test_loss', opts=dict(title='test_loss'), update='append')
        auc = roc_auc_score(all_label, all_out)
        print(auc)
        # print(all_out)

        # thre, _ = get_yoden_threshold(all_label, all_out)
        # print('Threshold: %.03f' % (thre))
        # acc, Sen, Spe = cal_acc(0.415, all_out, all_label)
        # print('ACC: %.03f | Sensitivity: %.03f | Specifity: %.03f ' % (acc, Sen, Spe))

        return all_out, all_label, all_name


if __name__ == '__main__':
    # starttime = datetime.datetime.now()
    test_out, label, all_name = my_test(1)
    # long running
    # endtime = datetime.datetime.now()
    # print('time')
    # print((endtime - starttime).seconds)

    dict_info = {"name": all_name,"label": label.tolist(), "prob": test_out.tolist()}
    print(test_out)
    print(type(test_out))

    info = pd.DataFrame(dict_info)

    file_name = 'DL_prob.xlsx'
    file_save_path = r'E:\Radiomics\huaxi_jiang_yinjie\outcome\prob'
    file_save_path = os.path.join(file_save_path, file_name)
    writer = pd.ExcelWriter(file_save_path)
    info.to_excel(writer)
    writer.save()

    out = test_out
    auc = roc_auc_score(label, out)
    thred = 0.428
    out[out > thred] = 1
    out[out <= thred] = 0
    acc = accuracy_score(label, out)
    print('acc')
    print(acc)
    print('auc')
    print(auc)

