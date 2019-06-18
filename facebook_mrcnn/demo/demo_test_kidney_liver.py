import os

import cv2
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from facebook_mrcnn.demo.predictor_kidney import KidneyDemo
from facebook_mrcnn.maskrcnn_benchmark.config import cfg


def split_path(file_path):
    path, name_ext = os.path.split(file_path)
    name, ext = os.path.splitext(name_ext)

    return path, name, ext


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

    print(path)
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


def image_dict(path, exts=['.png', '.jpg'], recursive=True, key=None, followlinks=True):
    path = norm_path(path)

    if key == None:
        key = lambda p: os.path.splitext(os.path.split(p)[-1])[0]

    d = dict()
    if recursive:
        for (root, dirs, files) in os.walk(path, followlinks=followlinks):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext.lower() not in exts:
                    continue

                full_path = os.path.join(root, file)
                d[key(full_path)] = full_path
    else:
        for fileDir in os.listdir(path):
            if os.path.isfile(os.path.join(path, fileDir)):
                file = fileDir
            else:
                continue

            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue

            full_path = os.path.join(path, file)
            d[key(full_path)] = full_path
    return d


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


def compute_iou(img1, img2):
    area1, area2 = img1.astype(np.bool), img2.astype(np.bool)

    area_inter = np.logical_and(area1, area2)
    area_union = np.logical_or(area1, area2)

    iou = np.count_nonzero(area_inter) / np.count_nonzero(area_union)

    return iou


def get_class_name_by_path(seg_path, file_path):
    seg_path = norm_path(seg_path)
    file_path = norm_path(file_path)

    sep = os.path.sep
    c = file_path.replace(seg_path, '').split(sep)[1]
    return c


def mask_overlap(img, seg_paths, w=20):
    for p in seg_paths:
        if p:
            img_seg = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img_seg_color = np.zeros(img.shape, dtype=np.uint8)
            img_seg_color[:, :, 2] = img_seg
            img = cv2.addWeighted(img, float(100 - w) * 0.01, img_seg_color, float(w) * 0.01, 0)
    return img


def main(dir_path=None, config_file=None, model_file=None, save_dir=None, classes=None, seg_path=None):
    dir_path = norm_path(dir_path) if dir_path else None
    config_file = norm_path(config_file) if config_file else None
    model_file = norm_path(model_file) if model_file else None
    save_dir = norm_path(save_dir, makedirs=True) if save_dir else None
    seg_path = norm_path(seg_path)

    save_crop_dir = norm_path(os.path.join(save_dir, 'crop'), makedirs=True) if save_dir else None
    save_mask_dir = norm_path(os.path.join(save_dir, 'mask'), makedirs=True) if save_dir else None

    print('paths', save_dir, save_crop_dir, save_mask_dir)

    # this makes our figures bigger
    #pylab -> pyplot, numpy를 같이 불러오는 역할
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
        CATEGORIES=classes
    )

    # count image per gt class
    seg_clss_dict = dict()
    for class_nm in classes[1:]:
        seg_clss_dict[class_nm] = image_dict(os.path.join(seg_path, class_nm))

    result_dict = dict()
    eval_image_list = image_list(dir_path)
    for filePath in eval_image_list:

        # export only name
        _, name, _ = split_path(filePath)

        # read image
        print('file_path', filePath)
        img = cv2.imread(filePath, cv2.IMREAD_COLOR)

        # resize image for input size of model
        img, resize_info = resize_aspect_ratio(img, (512, 512))
        result, crops, masks, labels = kidney_demo.detection(img)
        print('model labels', labels)

        # restore size of image to original size
        result = resize_restore(result, resize_info)

        # overlap mask area
        mask_paths = list()
        for class_mask in seg_clss_dict.values():
            if name in class_mask:
                mask_paths.append(class_mask[name])
        if mask_paths:
            result = mask_overlap(result, mask_paths)

        # save result image
        if save_dir:
            save_file = os.path.join(save_dir, fileName(filePath))
            cv2.imwrite(save_file, result)
        else:
            # imshow(result)
            pass

        for class_num, class_name in enumerate(classes):
            # get ai_class, ai_seg_file
            if class_num in labels:
                result_index = list(labels).index(class_num)

                # save ai seg image
                ai_seg_img = masks[result_index]
                ai_class_name = classes[labels[result_index]]
                save_path = os.path.join(save_mask_dir, ai_class_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file = os.path.join(save_path, fileName(filePath))
                ai_seg_img = resize_restore(ai_seg_img, resize_info)
                cv2.imwrite(save_file, ai_seg_img)

                ai_class = labels[result_index]
                ai_seg_file = save_file
            else:
                ai_class = 0  # None
                ai_seg_file = None

            # get gt_class, gt_seg_file
            if class_name in seg_clss_dict.keys() and name in seg_clss_dict[class_name]:
                gt_class = class_num
                gt_seg_file = seg_clss_dict[class_name][name]
            else:
                gt_class = 0  # __background__
                gt_seg_file = None

            # store result
            if name not in result_dict:
                result_dict[name] = list()

            if ai_class or gt_class:
                result_dict[name].append({'gt_class': gt_class, 'gt_seg_file': gt_seg_file,
                                          'ai_class': ai_class, 'ai_seg_file': ai_seg_file})

    # show_evaluation(result_dict, classes)
    show_evaluation_v2(result_dict, classes)


def show_evaluation(result_dict, classes):
    print('-' * 50)
    # for debug
    print('result_dict', result_dict)
    print('classes', classes)

    acc_class_list = dict()
    iou_class_list = dict()

    for file_nm, result_list in result_dict.items():
        for result in result_list:
            ai_class = result['ai_class']
            ai_seg_file = result['ai_seg_file']
            gt_class = result['gt_class']
            gt_seg_file = result['gt_seg_file']
            gt_class_nm = classes[gt_class]

            if ai_seg_file and gt_seg_file:
                gt_seg_img = cv2.imread(gt_seg_file, cv2.IMREAD_GRAYSCALE)
                ai_seg_img = cv2.imread(ai_seg_file, cv2.IMREAD_GRAYSCALE)
                iou = compute_iou(gt_seg_img, ai_seg_img)
            else:
                iou = 0.0

            if gt_class_nm not in acc_class_list:
                acc_class_list[gt_class_nm] = list()
            if gt_class_nm not in iou_class_list:
                iou_class_list[gt_class_nm] = list()

            if ai_class == gt_class and iou >= 0.5:
                acc_class_list[gt_class_nm].append((file_nm, True))
                iou_class_list[gt_class_nm].append((file_nm, iou))
            else:
                acc_class_list[gt_class_nm].append((file_nm, False))
                # iou_class_list[gt_class_nm].append(iou)

    all_class_cnt_correct = 0
    all_class_cnt_wrong = 0
    for class_nm, file_acc_list in acc_class_list.items():
        acc_list = [file_acc[1] for file_acc in file_acc_list]
        cnt_correct = sum([1 for acc in acc_list if acc])
        cnt_wrong = sum([1 for acc in acc_list if not acc])
        acc = cnt_correct / (cnt_correct + cnt_wrong)

        all_class_cnt_correct += cnt_correct
        all_class_cnt_wrong += cnt_wrong

        print('{} 적중:{:.2f}, cnt_correct:{}, cnt_wrong:{}'.format(class_nm, acc * 100, cnt_correct, cnt_wrong))
    all_class_acc = all_class_cnt_correct / (all_class_cnt_correct + all_class_cnt_wrong)
    print('분류 정확도:{:.1f}, cnt_correct:{}, cnt_wrong:{}'.format류(all_class_acc * 100, all_class_cnt_correct,
                                                               all_class_cnt_wrong))
    print()

    all_class_iou_list = list()
    for class_nm, file_iou_list in iou_class_list.items():
        iou_list = [file_iou[1] for file_iou in file_iou_list]
        if iou_list:
            meanIOU = sum(iou_list) / len(iou_list)
            maxIOU = max(iou_list)
            minIOU = min(iou_list)
        else:
            meanIOU = 0.0
            maxIOU = 0.0
            minIOU = 0.0

        all_class_iou_list.extend(iou_list)
        print('{} 적중도 meanIOU:{:.2f}, maxIOU:{:.2f}, meanIOU:{:.2f}'.format(class_nm, meanIOU, maxIOU, minIOU))
    meanIOU = sum(all_class_iou_list) / len(all_class_iou_list)
    maxIOU = max(all_class_iou_list)
    minIOU = min(all_class_iou_list)
    print('세그먼테이션 정확도 meanIOU:{:.2f}, maxIOU:{:.2f}, meanIOU:{:.2f}'.format(meanIOU, maxIOU, minIOU))


def show_evaluation_v2(result_dict, classes):
    print('-' * 50)
    # for debug
    print('result_dict', result_dict)
    print('classes', classes)

    # result_dict[name].append({'gt_class': gt_class, 'gt_seg_file': gt_seg_file,
    #                           'ai_class': ai_class, 'ai_seg_file': ai_seg_file})

    acc_class_list = dict()  # {class: [(file, T/F), ...], ...}
    iou_class_list = dict()  # {class: [(file, IOU), ...], ...}
    mistake_bg_for_other = dict()  # {class: [file, ...], ...}
    for class_nm in classes:
        if class_nm != '__background__':
            mistake_bg_for_other[class_nm] = list()

    for file_nm, result_list in result_dict.items():
        print('result_list', file_nm, result_list)
        for result in result_list:
            ai_class = result['ai_class']
            ai_class_nm = classes[ai_class]
            ai_seg_file = result['ai_seg_file']
            gt_class = result['gt_class']
            gt_seg_file = result['gt_seg_file']
            gt_class_nm = classes[gt_class]

            if ai_seg_file and gt_seg_file:
                gt_seg_img = cv2.imread(gt_seg_file, cv2.IMREAD_GRAYSCALE)
                ai_seg_img = cv2.imread(ai_seg_file, cv2.IMREAD_GRAYSCALE)
                iou = compute_iou(gt_seg_img, ai_seg_img)
            else:
                iou = 0.0

            if gt_class_nm not in acc_class_list:
                acc_class_list[gt_class_nm] = list()
            if gt_class_nm not in iou_class_list:
                iou_class_list[gt_class_nm] = list()

            if ai_class == gt_class and iou >= 0.5:
                acc_class_list[gt_class_nm].append((file_nm, True))
                iou_class_list[gt_class_nm].append((file_nm, iou))
            else:
                acc_class_list[gt_class_nm].append((file_nm, False))
                # iou_class_list[gt_class_nm].append(iou)

            if gt_class_nm == '__background__' and ai_class_nm != '__background__':
                mistake_bg_for_other[ai_class_nm].append(file_nm)

    all_class_cnt_correct = 0
    all_class_cnt_wrong = 0
    for class_nm, file_acc_list in acc_class_list.items():
        acc_list = [file_acc[1] for file_acc in file_acc_list]
        cnt_correct = sum([1 for acc in acc_list if acc])
        cnt_wrong = sum([1 for acc in acc_list if not acc])
        acc = cnt_correct / (cnt_correct + cnt_wrong)

        all_class_cnt_correct += cnt_correct
        all_class_cnt_wrong += cnt_wrong

        print('{} 적중:{:.2f}, cnt_correct:{}, cnt_wrong:{}'.format(class_nm, acc * 100, cnt_correct, cnt_wrong))
    all_class_acc = all_class_cnt_correct / (all_class_cnt_correct + all_class_cnt_wrong)
    print('분류 정확도:{:.1f}, cnt_correct:{}, cnt_wrong:{}'.format(all_class_acc * 100, all_class_cnt_correct,
                                                               all_class_cnt_wrong))
    print()

    print('__background__로 오인', )
    for class_nm in mistake_bg_for_other.keys():
        print('\t{}: {}', class_nm, len(mistake_bg_for_other[class_nm]))
    print()

    all_class_iou_list = list()
    for class_nm, file_iou_list in iou_class_list.items():
        iou_list = [file_iou[1] for file_iou in file_iou_list]
        if iou_list:
            meanIOU = sum(iou_list) / len(iou_list)
            maxIOU = max(iou_list)
            minIOU = min(iou_list)
        else:
            meanIOU = 0.0
            maxIOU = 0.0
            minIOU = 0.0

        all_class_iou_list.extend(iou_list)
        print('{} 적중도 meanIOU:{:.2f}, maxIOU:{:.2f}, meanIOU:{:.2f}'.format(class_nm, meanIOU, maxIOU, minIOU))
    meanIOU = sum(all_class_iou_list) / len(all_class_iou_list)
    maxIOU = max(all_class_iou_list)
    minIOU = min(all_class_iou_list)
    print('세그먼테이션 정확도 meanIOU:{:.2f}, maxIOU:{:.2f}, meanIOU:{:.2f}'.format(meanIOU, maxIOU, minIOU))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config_file = "../configs/kidney/e2e_mask_rcnn_X_101_32x8d_FPN_1x_liver_kidney_augpolygon_using_pretrained_model.yaml"
    dir_path = "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527/us/val"
    model_file = "/home/bong6/lib/robin_mrcnn/facebook_mrcnn/checkpoint1/liver_kidney_augpolygon_using_pretrained_model/model_0028000.pth"
    seg_path = "/home/bong6/data/SegMrcnn_kidney_liver_400_100_20190527/seg"
    save_dir = '/home/bong6/data/train_result/liver_kidney_classchange'

    classes = ['__background__', 'liver', 'kidney']
    # classes = ['__background__', 'liver']

    main(dir_path=dir_path, config_file=config_file, model_file=model_file, save_dir=save_dir, classes=classes,
         seg_path=seg_path)
