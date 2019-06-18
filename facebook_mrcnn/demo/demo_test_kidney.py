import os

import cv2
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from facebook_mrcnn.demo.predictor_kidney import KidneyDemo
from facebook_mrcnn.maskrcnn_benchmark.config import cfg


def load(path):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(path).convert("RGB")

    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.waitforbuttonpress()


def norm_path(path, makedirs=False):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if makedirs and not os.path.exists(path):
        os.makedirs(path)

    return path


def fileName(filePath):
    dir, fileNameExt = os.path.split(filePath)

    return fileNameExt


def image_list(path, exts=['.png', '.jpg'], recursive=True, followlinks=True):
    path = norm_path(path)

    l = list()
    if recursive:
        for (root, dirs, files) in os.walk(path, followlinks=followlinks):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext.lower() not in exts:
                    continue

                l.append(os.path.join(root, file))
    else:
        for fileDir in os.listdir(path):
            if os.path.isfile(os.path.join(path, fileDir)):
                file = fileDir
            else:
                continue

            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue

            l.append(os.path.join(path, file))
    return l


def resize_aspect_ratio(img, sizeWH):
    resize_info = {'original_size': [None, None],
                   'pad_top_bottom_left_right': [None, None, None, None],
                   'windowXYWH': [None, None, None, None]}

    dw, dh = sizeWH
    h, w = img.shape[:2]
    resize_info['original_size'] = [w, h]
    ratio = min([sizeWH[1] / h, sizeWH[0] / w])

    nw, nh = (int(w * ratio), int(h * ratio))
    img = cv2.resize(img, (nw, nh))

    padL, padT = (dw - nw) // 2, (dh - nh) // 2
    padR, padB = dw - nw - padL, dh - nh - padT
    resize_info['pad_top_bottom_left_right'] = [padT, padB, padL, padR]
    resize_info['windowXYWH'] = [padL, padT, nw, nh]

    img = cv2.copyMakeBorder(img, padT, padB, padL, padR, borderType=cv2.BORDER_CONSTANT, value=0)

    return img, resize_info


def resize_restore(img, resize_info):
    xw, yw, ww, hw = resize_info['windowXYWH']
    img = img[yw:yw + hw, xw:xw + ww]

    w, h = resize_info['original_size']
    img = cv2.resize(img, (w, h))

    return img


def main(dir_path=None, config_file=None, model_file=None, save_dir=None):
    dir_path = norm_path(dir_path) if dir_path else None
    config_file = norm_path(config_file) if config_file else None
    model_file = norm_path(model_file) if model_file else None
    save_dir = norm_path(save_dir, makedirs=True) if save_dir else None
    save_crop_dir = norm_path(os.path.join(save_dir, 'crop'), makedirs=True) if save_dir else None
    save_mask_dir = norm_path(os.path.join(save_dir, 'mask'), makedirs=True) if save_dir else None

    print('paths', save_dir, save_crop_dir, save_mask_dir)

    # this makes our figures bigger
    pylab.rcParams['figure.figsize'] = 20, 12
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    cfg.merge_from_list(["MODEL.WEIGHT", model_file])

    kidney_demo = KidneyDemo(
        cfg,
        min_image_size=512,
        confidence_threshold=0.7,
    )

    kidney_label = 1

    for filePath in image_list(dir_path):

        print('file_path', filePath)
        img = cv2.imread(filePath, cv2.IMREAD_COLOR)

        # resize image for input size of model
        img, resize_info = resize_aspect_ratio(img, (512, 512))
        result, crops, masks, labels = kidney_demo.detection(img)

        # restore size of image to original size
        result = resize_restore(result, resize_info)

        # save result image
        if save_dir:
            save_file = os.path.join(save_dir, fileName(filePath))
            cv2.imwrite(save_file, result)
        else:
            imshow(result)

        # if found object, make corp and mask image
        if len(labels) > 0:
            for crop, mask, label in zip(crops, masks, labels):
                if label != kidney_label:
                    continue

                if save_crop_dir:
                    save_file = os.path.join(save_crop_dir, fileName(filePath))
                    crop = resize_restore(crop, resize_info)
                    cv2.imwrite(save_file, crop)

                if save_mask_dir:
                    save_file = os.path.join(save_mask_dir, fileName(filePath))
                    mask = resize_restore(mask, resize_info)
                    cv2.imwrite(save_file, mask)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x_kidney.yaml"
    # dir_path = "/home/bong07/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/val"
    # model_file = "/home/bong07/lib/robin_mrcnn/checkpoint/20190121-164312/model_0165000.pth"
    # save_dir = '../result/20190118-175358_model_0165000'

    config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x_kidney.yaml"
    dir_path = "/home/bong07/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/val"
    model_file = "/home/bong07/lib/robin_mrcnn/checkpoint/20190123-195649/model_0105000.pth"
    save_dir = '../result/20190123-195649_model_0105000'



    main(dir_path=dir_path, config_file=config_file, model_file=model_file, save_dir=save_dir)
