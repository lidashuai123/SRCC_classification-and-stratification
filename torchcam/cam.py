from .saliency import get_image_saliency_result, get_image_saliency_plot

__all__ = ['getCAM']


def getCAM(model, raw_image, input, label, ID, layer_path=None, display=True, save=False):
    saliency_maps = get_image_saliency_result(model, raw_image, input, label,
                                              methods=['gradcam'], layer_path=layer_path)
    # print(saliency_maps)
    gray_img, color_img = get_image_saliency_plot(saliency_maps, ID, display=display, save=save)

    return gray_img, color_img
