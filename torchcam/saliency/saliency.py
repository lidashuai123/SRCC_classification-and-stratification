import torch
import math
import skimage.transform
import matplotlib.pyplot as plt

from .explainer import get_explainer
from ..helper import show_image
from PIL import Image
import cv2
import matplotlib.cm as cm

__all__ = ['get_image_saliency_result', 'get_image_saliency_plot']


class SaliencyImage(object):
    def __init__(self, raw_image, saliency, title, saliency_alpha=0.5, saliency_cmap='jet'):
        self.raw_image, self.saliency, self.title = raw_image, saliency, title
        self.saliency_alpha, self.saliency_cmap = saliency_alpha, saliency_cmap


def get_saliency(model, raw_input, input, label, method='gradcam', layer_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    input = input.to(device)
    if label is not None:
        label = label.to(device)

    if input.grad is not None:
        input.grad.zero_()
    if label is not None and label.grad is not None:
        label.grad.zero_()
    model.eval()
    model.zero_grad()

    exp = get_explainer(method, model, layer_path)

    saliency = exp.explain(input, label, raw_input)
    # print(layer_path)

    if saliency is not None:
        saliency = saliency.abs().sum(dim=1)[0].squeeze()
        saliency -= saliency.min()
        saliency /= (saliency.max() + 1e-20)
        return saliency.detach().cpu().numpy()
    else:
        return None


# TODO: support lime_imagenet, guided_backprop, deeplift, etc.
def get_image_saliency_result(model, raw_image, input, label,
                              methods=['gradcam'],##'smooth_grad',, 'vanilla_grad', 'grad_x_input'
                              layer_path=None):
    result = list()
    for method in methods:
        sal = get_saliency(model, raw_image, input, label, method=method, layer_path=layer_path)
        # print(sal)
        if sal is not None:
            result.append(SaliencyImage(raw_image, sal, method))

    return result


def get_image_saliency_plot(image_saliency_results, ID, cols: int = 2, figsize: tuple = None, display=True, save=False):
    ##直接保存
    for i, r in enumerate(image_saliency_results):
        # print(type(r.saliency))
        # print(r.saliency.shape)
        # print(r.saliency_cmap)
        # print(r.saliency_alpha)

        saliency_upsampled = skimage.transform.resize(r.saliency,
                                                      (r.raw_image.height, r.raw_image.width),
                                                      mode='reflect')
        # cmap = cm.jet_r(saliency_upsampled)[..., :1] * 255.0
        # print(saliency_upsampled.max())
        saliency_upsampled = (saliency_upsampled*255).astype('uint8')
        im_color = cv2.applyColorMap(saliency_upsampled, cv2.COLORMAP_JET)
        # print(im_color.shape)

        # cv2.imshow('licong', im_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.destroyWindow('licong')

    # rows = math.ceil(len(image_saliency_results) / cols)
    # figsize = figsize or (8, 3 * rows)
    # figure = plt.figure(figsize=figsize)
    #
    # for i, r in enumerate(image_saliency_results):
    #     ax = figure.add_subplot(rows, cols, i + 1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title(r.title, fontdict={'fontsize': 18})
    #
    #     saliency_upsampled = skimage.transform.resize(r.saliency,
    #                                                   (r.raw_image.height, r.raw_image.width),
    #                                                   mode='reflect')
    #
    #     show_image(r.raw_image, img2=saliency_upsampled, alpha2=r.saliency_alpha, cmap2=r.saliency_cmap, ax=ax)
    #
    # if display:
    #     figure.show()
    #     figure.waitforbuttonpress()
    #     # plt.close()
    # if save:
    #     figure.savefig('E:\\data\\BraTs\\BraTs_cam\\%s_cam.png'%ID)
    # plt.close()

    return saliency_upsampled, im_color
