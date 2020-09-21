"""
此文件用于写训练的过程
"""
# @Author   : Cong
# @Time : ${DATE} ${TIME}

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
from sklearn.metrics import roc_auc_score
from visdom import Visdom

from my_mobilenet import mobilenet_v2
from my_resnet import my_resnet18, resnet34, resnet50
from Unet_network import ResUNetEncoder


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
    T.RandomRotation(45),
    T.ToTensor(),
    # T.Normalize(mean=[0.414, 0.348, 0.460],
    #             std=[0.226, 0.188, 0.250])
    # T.Normalize(mean=[0.254, 0.213, 0.282],
    #             std=[0.235, 0.197, 0.264])
    T.Normalize(mean=[0.459],
                std=[0.250])
])

valid_transforms = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    # T.Normalize(mean=[0.414, 0.348, 0.460],
    #             std=[0.226, 0.188, 0.250])
    # T.Normalize(mean=[0.254, 0.213, 0.282],
    #             std=[0.235, 0.197, 0.264])
    T.Normalize(mean=[0.459],
                std=[0.250])
])

use_gpu = torch.cuda.is_available()
print(use_gpu)

##固定窗口大小的resize为112  外接矩形的resize为64
file_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\fixed_rec_cropped_images.xlsx'
# file_path = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\pure_rec_cropped_images.xlsx'

train_dataset = Yinjie_data(
    file_path, transforms=train_transforms, train=True,
    test=False)
# valid_dataset = Yinjie_data(
#     file_path, transforms=valid_transforms, train=True,
#     test=True)
test_dataset = Yinjie_data(
    file_path, transforms=valid_transforms, train=False,
    test=True)

batchsize = 64

# weights = [2 if label == 1 else 1 for data, label in train_dataset]
# sampler = WeightedRandomSampler(weights=weights, num_samples=1000, replacement=True)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=0)
# validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)

##resnet网络结构
# model = models.resnet18()
# # model = models.resnet18()
# fc_features = model.fc.in_features
# model.fc = nn.Linear(fc_features, 2)
# conv1_out_channels = model.conv1.out_channels
# model.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=7, stride=2, padding=3,
#                                bias=False)

##加载自己更改的resnet结构,加载预训练权重
# model = my_resnet18()
model = ResUNetEncoder(in_ch=1, out_ch=2)

# pretrained_weight_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\seg_model\base_99.pth.tar'
# pretrained_dict = torch.load(pretrained_weight_path)
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# if len(pretrained_dict) != 0:
#     print('加载成功')


pretrained_weight_path = r'E:\pycharm_project\Huaxi_Yinjie_Jiang\Unet_encoder_weights\epoch_101_75_72.pth'
pretrained_dict = torch.load(pretrained_weight_path)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
if len(pretrained_dict) != 0:
    print('加载成功')


##mobilenet网络
# model = mobilenet_v2()

##参数一半的densenet121网络
# model = models.densenet.densenet121_half()
# classifier_num_features = model.classifier.in_features
# model.classifier = nn.Linear(classifier_num_features, 2)
# conv1_out_channels = model.features.conv0.out_channels
# model.features.conv0 = nn.Conv2d(1, conv1_out_channels, kernel_size=7, stride=2,
#                                 padding=3, bias=False)

##inception网络
# model = models.inception_v3()
# classifier_num_features = model.fc.in_features
# model.fc = nn.Linear(classifier_num_features, 2)
# model_conv1a_out_channels = model.Conv2d_1a_3x3.conv.out_channels
# print(model_conv1a_out_channels)
# model.Conv2d_1a_3x3.conv = nn.Conv2d(1, model_conv1a_out_channels, bias=False, kernel_size=3, stride=2)
# model.aux_logits = False


model = model.cuda()

criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
lr = 1e-6
weight_decay = 10
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)##用10不错
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=0.9)

# scheduler = StepLR(optimizer, gamma=0.2, step_size=10)##先用固定学习率 gamma是学习率衰减的倍数
criterion.cuda()
epoch = 220


def train(epoch):
    print('\nEpoch: %d' % epoch)
    # scheduler.step()
    model.train()
    all_label = np.array([])
    all_out = np.array([])
    for batch_idx, (img, label, _) in enumerate(trainloader):
        image = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        out = model(image)
        # out = nn.Sigmoid()(out)###计算BCEloss
        # print(out)
        ##计算auc
        ##计算auc 使用crossentropy损失时
        out_array = out.data.cpu().numpy()
        out_prob = my_softmax(out_array)
        out_auc = out_prob[:, 1]  ##取出其中的一列用于计算auc
        all_out = np.hstack([all_out, out_auc])
        label_array = label.cpu().numpy()
        all_label = np.hstack([all_label, label_array])
        ###计算BCEloss损失时
        # out_auc = out.data.cpu().numpy()[:, 0]
        # all_out = np.hstack([all_out, out_auc])
        # label_array = label.cpu().numpy()
        # all_label = np.hstack([all_label, label_array])

        # import ipdb
        # ipdb.set_trace()
        loss = criterion(out, label)
        # print(loss)
        loss.backward()
        optimizer.step()
        # print(label)
        # print(out)
        #print("Epoch:%d [%d|%d] loss:%f" % (epoch, batch_idx, len(trainloader), loss.item()))
        viz.line([loss.item()], [global_step*len(trainloader) + batch_idx], win='train_loss', opts=dict(title='loss'), update='append')
    auc = roc_auc_score(all_label, all_out)

    print("Epoch:%d  loss:%f   auc:%f" % (epoch, loss.item(), auc))
    return auc


def my_test(epoch):
    print("\nTest Epoch: %d" % epoch)
    model.eval()
    with torch.no_grad():
        all_label = np.array([])
        all_out = np.array([])
        for batch_idx, (img, label, _) in enumerate(testloader):
            image = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(image)
            # out = nn.Sigmoid()(out)

            ##计算auc
            ##使用Crossentropy损失时
            out_array = out.data.cpu().numpy()
            out_prob = my_softmax(out_array)
            out_auc = out_prob[:, 1]  ##取出其中的一列用于计算auc
            all_out = np.hstack([all_out, out_auc])
            label_array = label.cpu().numpy()
            all_label = np.hstack([all_label, label_array])
            ##使用BCEloss损失时
            # out_auc = out.data.cpu().numpy()[:, 0]
            # all_out = np.hstack([all_out, out_auc])
            # label_array = label.cpu().numpy()
            # all_label = np.hstack([all_label, label_array])

            loss = criterion(out, label)
            # print("Valida__Epoch:%d [%d|%d] loss:%f， auc:%f" % (epoch, batch_idx, len(trainloader), loss.item(), auc))
            viz.line([loss.item()], [global_step * len(testloader)+ batch_idx], win='test_loss', opts=dict(title='test_loss'), update='append')
        auc = roc_auc_score(all_label, all_out)
        # print(all_label)
        # print(all_out)
        print("Test__Epoch:%d loss:%f， auc:%f" % (epoch, loss.item(), auc))
        return auc


if __name__ == '__main__':

    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='loss'))
    viz.line([0.], [0.], win='test_loss', opts=dict(title='test_loss'))
    viz.line([[0.0, 0.0]], [0.], win='auc', opts=dict(title='auc',
                                                       legend=['train_auc', 'test_auc']))

    viz.text(f"学习率:{lr }       weight_decay:{weight_decay}      model_type: my_resnet18", win='info',
             opts=dict(title='info'))

    global_step = 0
    metrics = []
    save_path = r'E:\pycharm_project\Huaxi_Yinjie_Jiang\Unet_encoder_weights'
    for epoch in range(epoch):
        global_step += 1
        train_auc = train(epoch)
        # valid(epoch)
        test_auc = my_test(epoch)
        viz.line([[train_auc, test_auc]], [global_step], win='auc', opts=dict(title='auc', legend=['train_auc', 'test_auc']), update='append')

        metric = test_auc - abs(train_auc-test_auc) * 0.5
        metrics.append(metric)

        model_name = 'epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), os.path.join(save_path, model_name))
        if metric >= max(metrics) and 0.69 < test_auc < train_auc + 0.2:
            #model_name = 'noStepLr'+ str(lr) + '-' + str(weight_decay) + '-' + 'my_resnet18' + '-' + str(epoch) + '-' + 'nofixed' + '-' + 'classifier' + '.pth'
            #torch.save(model.state_dict(), os.path.join(save_path, model_name))
            viz.text(f"epoch:{epoch},name:{model_name}:::train_auc:{train_auc}, test_auc:{test_auc}", win='info',
                     opts=dict(title='info'), append=True)



