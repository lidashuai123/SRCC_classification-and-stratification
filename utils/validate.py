from Unet_dataset import Promise_data
from torch.utils.data import DataLoader
from Unet_network import ResUNet
import torch
import torch.nn as nn
import os
import numpy as np
from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding
from torchvision.transforms import ToTensor,Compose
from PIL import Image
from torchvision.transforms import ToPILImage

def save_picture(pic, new_id, mask=True, generate=True):
    ###linux
    # save_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\out'##PVP
    save_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\AP_out'##AP

    if mask and not generate:###金标准
        # print('groundtruth')
        # print(pic.shape)
        # print(pic.max())
        pic = pic * 255
        im = Image.fromarray(np.uint8(pic), mode='L')
        im.save(os.path.join(save_path, str(new_id)+'ground_truth.png'))
    elif mask and generate:##分割网络结果
        # print('predict')
        # print(pic.shape)
        # print(pic.max())
        pic = pic * 255
        im = Image.fromarray(pic, mode='L')
        im.save(os.path.join(save_path, str(new_id)+'generater.png'))
    else:##原始图片
        # print(pic.shape)
        # pic = pic.squeeze((0, 1))
        # print(pic.shape)
        # print(pic.min())
        # pic = pic * 255
        # im = Image.fromarray(pic, mode='L')
        im = ToPILImage()(pic)
        im.save(os.path.join(save_path, str(new_id)+ 'img.png'))


def save_picture_unlabel(pic, new_id, mask=True, generate=True):
    ###linux
    # save_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\out'##PVP
    save_path = r'E:\Radiomics\huaxi_jiang_yinjie\segmentation\AP_out'##AP

    if mask and not generate:###金标准
        # print('groundtruth')
        # print(pic.shape)
        # print(pic.max())
        pic = pic * 255
        im = Image.fromarray(np.uint8(pic), mode='L')
        im.save(os.path.join(save_path, str(new_id)+'ground_truth.png'))
    elif mask and generate:##分割网络结果
        # print('predict')
        # print(pic.shape)
        # print(pic.max())
        pic = pic * 255
        im = Image.fromarray(pic, mode='L')
        im.save(os.path.join(save_path, str(new_id)+'generater.png'))
    else:##原始图片
        # print(pic.shape)
        # pic = pic.squeeze((0, 1))
        # print(pic.shape)
        # print(pic.min())
        # pic = pic * 255
        # im = Image.fromarray(pic, mode='L')
        im = ToPILImage()(pic)
        im.save(os.path.join(save_path, str(new_id)+'img.png'))


def val(model,valoader,epoch, nclass=2,use_cuda=True):
    model.eval()
    gts, preds = [], []
    for img_id, (img,gt_mask,_, new_id) in enumerate(valoader):
        # print(img.shape)
        # print(gt_mask.shape)
        # save_picture(img[0], slice_id, mask=False, generate=False)
        gt_mask = gt_mask.numpy()[0]
        # print(f'gt_mask.shape:{gt_mask.shape}')
        if use_cuda:
            img = img.cuda()
        else:
            img = img
        out_pred_map = model(img)
        # Get hard prediction
        if use_cuda:
            soft_pred = out_pred_map.data.cpu().numpy()[0]
            # print(soft_pred.shape)
        else:
            soft_pred = out_pred_map.data.numpy()[0]

        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)

        # gts.append(gt_mask)
        # preds.append(hard_pred)
        # if epoch % 20 == 0:
        #     save_picture(hard_pred, new_id, mask=True, generate=True)
        #     save_picture(gt_mask, new_id, mask=True, generate=False)

        gts.append(gt_mask)
        preds.append(hard_pred)

        # print(gt_mask.shape())
        # for gt_, pred_, id in zip(gt_mask, hard_pred, list(new_id)):
            # gts.append(gt_mask)
            # preds.append(hard_pred)
        if epoch % 20 == 0:
            save_picture(hard_pred, new_id, mask=True, generate=True)
            save_picture(gt_mask, new_id, mask=True, generate=False)

    _, miou = scores(gts, preds, n_class = nclass)

    return miou

def val_unlabel(model,trainloder_u, epoch, nclass=2,use_cuda=True):
    model.eval()
    gts, preds = [], []
    for img_id, (img,gt_mask,_, new_id) in enumerate(trainloder_u):
        # print(img.shape)
        # print(gt_mask.shape)
        # save_picture(img[0], slice_id, mask=False, generate=False)
        gt_mask = gt_mask.numpy()
        # print(f'gt_mask.shape:{gt_mask.shape}')
        if use_cuda:
            img = img.cuda()
        else:
            img = img
        out_pred_map = model(img)
        # Get hard prediction
        if use_cuda:
            soft_pred = out_pred_map.data.cpu().numpy()
            # print(soft_pred.shape)
        else:
            soft_pred = out_pred_map.data.numpy()

        soft_pred = soft_pred[:,:, :gt_mask.shape[1],:gt_mask.shape[2]]
        hard_pred = np.argmax(soft_pred,axis=1).astype(np.uint8)

        gts.append(gt_mask)
        preds.append(hard_pred)
        for gt_, pred_, id in zip(gt_mask, hard_pred, list(new_id)):
            # print(f'gt_shape:{gt_.shape}')
            # gts.append(gt_mask)
            # preds.append(hard_pred)
            if epoch % 20 == 0:
                save_picture_unlabel(pred_, id, mask=True, generate=True)
                save_picture_unlabel(gt_, id, mask=True, generate=False)

    _, miou = scores(gts, preds, n_class = nclass)

    return miou

if __name__ == '__main__':
    a = np.array([[0,0,0,0],[0,1,1,0],[0,0,1,0], [0,0,0,0]])
    b = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    _, dic = scores(a, b, n_class=2)
    print(dic)
