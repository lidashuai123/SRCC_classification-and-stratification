
from Unet_network import ResUNet
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import random
from torchvision.transforms import ToTensor, Compose, CenterCrop, Resize, Normalize
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding
from Unet_dataset import Promise_data
import os
from utils.lr_scheduling import poly_lr_scheduler
from utils.validate import val, val_unlabel
import numpy as np
import json

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mode = 'base'
pretrain = False
# f_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\log\log1.json'##PVP
f_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\log\log_AP.json'##AP


###超参数设置
use_cuda = True
no_norm = True###是否自己标准化
batch_size = 8
val_orig = False###是否使用原图验证
g_lr = 0.0001###原始是0.00025
d_lr = 0.00001
# g_lr = 0###用于测试生成图片
# d_lr = 0
d_optim = 'adam'
start_epoch = 1
max_epoch = 500
prefix = mode

d_label_smooth = 0###相当于正则化，等于零的时候没有效果
lam_semi = 0#0.01
lam_adv = 0.01###原本是0.01
wait_semi = 10###等待半监督开始的epoch,原始10
t_semi = 0.2###设置置信阈值

home_dir = r'E:\Radiomics\huaxi_jiang_yinjie\cropped_images\image_roi_path_file\Unet_path.xlsx'###V1
# snapshot_dir = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\seg_model'###PVP
snapshot_dir = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\AP_seg_model'##AP

# ###windows
# home_dir = r'E:\data\PROMISE12\path_file\images_segs_path.xlsx'
# f_path = r'E:\pycharm_project\my_promise\log\log2.json'
# snapshot_dir = r'E:\data\PROMISE12\model_save'

'''
    Snapshot the Best Model
'''
def snapshot(model,optimG, trainloader_l, valoader,epoch,best_miou_unlabel, best_miou,snapshot_dir,prefix, f_path):

    miou_unlabel = val_unlabel(model, trainloader_l, epoch)
    miou = val(model, valoader, epoch)

    global best_epoch
    global best_epoch_unlabel
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimG.state_dict(),
        'miou_unlabel': miou_unlabel,
        'miou': miou
    }
    if miou_unlabel > best_miou_unlabel:
        best_miou_unlabel = miou_unlabel
        best_epoch_unlabel = epoch
        torch.save(snapshot,os.path.join(snapshot_dir,'{}.pth.tar'.format(prefix)))
    if miou > best_miou:
        best_miou = miou
        best_epoch = epoch
        torch.save(snapshot, os.path.join(snapshot_dir, '{}.pth.tar'.format(prefix)))

    print("unlabelled::[{}] Curr mIoU: {:0.4f} Best mIoU: {} Best epoch: {}".format(epoch,miou_unlabel,best_miou_unlabel, best_epoch_unlabel))
    print("validation::[{}] Curr mIoU: {:0.4f} Best mIoU: {} Best epoch: {}".format(epoch, miou, best_miou, best_epoch))
    with open(f_path, 'a') as f:
        f.write("unlabelled::[{}] Curr mIoU: {:0.4f} Best mIoU: {} Best epoch: {}".format(epoch,miou_unlabel,best_miou_unlabel, best_epoch_unlabel))
        f.write("validation::[{}] Curr mIoU: {:0.4f} Best mIoU: {} Best epoch: {}".format(epoch, miou, best_miou, best_epoch))
        f.write('\n')

    return best_miou_unlabel, best_miou

def train_base(generator,optimG,trainloader, valoader, f_path):
    best_miou_unlabel = -1
    best_miou = -1
    for epoch in range(start_epoch, max_epoch+1):
        generator.train()
        for batch_id, (img, mask, _, _) in enumerate(trainloader):

            if not use_cuda:
                img,mask = img, mask
            else:
                img,mask = img.cuda(),mask.cuda()

            # print(mask.max())###测试mask的最大值是多少

            itr = len(trainloader)*(epoch-1) + batch_id
            cprob = generator(img)
            cprob = nn.LogSoftmax()(cprob)
            # print(cprob.shape)
            # print(mask.shape)

            Lseg = nn.NLLLoss()(cprob,mask)

            optimG = poly_lr_scheduler(optimG, g_lr, itr)
            optimG.zero_grad()

            Lseg.backward()
            optimG.step()

            print("[{}][{}]Loss: {:0.4f}".format(epoch,itr,Lseg.item()))

        best_miou_unlabel, best_miou = snapshot(generator,optimG, trainloader, valoader, epoch,best_miou_unlabel, best_miou,snapshot_dir, prefix + '_' + str(epoch), f_path)


def main():
    # args = parse_args()
    random.seed(0)
    torch.manual_seed(0)

    dict = {'home_dir': home_dir,'pretrain':str(pretrain),  'use_cuda': use_cuda, 'mode': mode, 'g_lr': g_lr, 'd_lr': d_lr, 'lam_semi': lam_semi,
            'lam_adv': lam_adv, 't_semi': t_semi,
            'batch_size': batch_size, 'wait_semi': wait_semi, 'd_optim': d_optim,
             'snapshot_dir': snapshot_dir, 'd_label_smooth': d_label_smooth,
            'val_orig': val_orig, 'start_epoch': start_epoch, 'max_epoch': max_epoch}

    json_str = json.dumps(dict, indent=4)
    with open(f_path, 'a') as f:
        f.write(json_str)
        f.write('\n')

    # normalize = Normalize(mean=[0.459], std=[0.250])##PVP
    normalize = Normalize(mean=[0.414], std=[0.227])  ##AP
    if use_cuda:
        torch.cuda.manual_seed_all(0)

    crop_size = 112
    imgtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)), ToTensor(),normalize]
    labtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)), ToTensorLabel()]

    cotr = []
    trainset_l = Promise_data(home_dir,img_transform=Compose(imgtr), label_transform=Compose(labtr),co_transform=Compose(cotr),labelled=True, valid=False)
    trainloader_l = DataLoader(trainset_l,batch_size=batch_size,shuffle=True,num_workers=0)

    #########################
    # Validation Dataloader #
    ########################
    if val_orig:##使用原图
        if no_norm:
            imgtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)),ToTensor(), normalize]
        else:
            imgtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)),ToTensor(),NormalizeOwn()]
        labtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)), ToTensorLabel()]
        cotr = []
    else:
        if no_norm:
            imgtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)),ToTensor(), normalize]
        else:
            imgtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)),ToTensor(),NormalizeOwn()]
        labtr = [CenterCrop((crop_size, crop_size)), Resize((112, 112)), ToTensorLabel()]
        # cotr = [RandomSizedCrop((112, 112))]
        cotr =[]

    # valset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), \
    #     label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)

    valset = Promise_data(home_dir, img_transform=Compose(imgtr), label_transform=Compose(labtr),
                              co_transform=Compose(cotr), labelled=True, valid=True)
    valoader = DataLoader(valset,batch_size=1)

    #############
    # GENERATOR #
    #############
    generator = ResUNet(in_ch=1, out_ch=2, base_ch=64)
    # if pretrain:
    #     pretrained_weight_path = '/data/lc/PROMISE12/model_save/semi_15.pth.tar'
    #     pretrained_dict = torch.load(pretrained_weight_path)
    #     generator.load_state_dict(pretrained_dict['state_dict'])
    #     if len(pretrained_dict) != 0:
    #         print('加载成功')

    # optimG = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),lr=g_lr,momentum=0.9,\
    #     weight_decay=0.0001,nesterov=True)
    optimG = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=g_lr, weight_decay=0.0001)
    if use_cuda:
        generator = generator.cuda()

    if mode == 'base':
        train_base(generator,optimG,trainloader_l, valoader, f_path)

if __name__ == '__main__':

    main()
