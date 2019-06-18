import os
import json
import cv2
import mmcv
import numpy as np

from mmcv.runner import load_checkpoint

from mmdetection.tools.visualize import visualize
from mmdetection.mmdet.core import get_classes
from mmdetection.mmdet.models import build_detector
from mmdetection.mmdet.apis import inference_detector


config_path = '/home/bong3/lib/robin_mrcnn/mmdetection/configs/dcn/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_kmu.py'
# config_path = '/home/bong3/lib/robin_mrcnn/mmdetection/configs/cascade_rcnn_r50_fpn_1x_rsna.py'
checkpoint_path = '/home/bong3/lib/robin_mrcnn/work_dirs/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_kmu/epoch_9.pth'
# checkpoint_path = '/home/bong3/lib/robin_mrcnn/mmdetection/tools/work_dirs/cascade_rcnn_r50_fpn_1x_rsna/latest.pth'

cfg = mmcv.Config.fromfile(config_path)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, checkpoint_path)

# test a list of images
# img_dir_path = '/home/bong3/data/rsna512/test'
# result_dir_path = '/home/bong3/data/rsna512/result'
# gt_ann_path = '/home/bong3/data/rsna512/test_ann.json'

img_dir_path = '/home/bong3/data/mmdetection_test/test'
result_dir_path = '/home/bong3/data/mmdetection_test/result'
gt_ann_path = '/home/bong3/data/mmdetection_test/kmu_resize_label_896_2048_2nd_mmd.json'

if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)

file_info_list = list()
with open(gt_ann_path, 'r') as f:
    file_info_list = json.load(f)

if len(file_info_list) > 0:
    img_info = dict()
    for file_info in file_info_list:
        filename = file_info['filename']
        gt_bboxes = file_info['ann']['bboxes']

        img_info[filename] = gt_bboxes

imgs = [os.path.join(img_dir_path, filename) for filename in os.listdir(img_dir_path)]

# dataset = 'rsna'
dataset = 'kmu'
score_thr=0.5
predictions = dict()

for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])

    # print(result)
    # mask 제외 시킨다 지금은
    result = result[0]

    file_path = imgs[i]
    filename = os.path.split(file_path)[-1]

    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(file_path)

    # if filename in img_info:
    #     gt_bboxes = img_info[filename]
    #
    #     for gt_bbox in gt_bboxes:
    #         img = cv2.rectangle(img, tuple(gt_bbox[:2]), tuple(gt_bbox[2:]), (255,0,0), 2)

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]

        inds = scores > score_thr

        scores = scores[inds]
        bboxes = bboxes[inds, :-1]
        labels = labels[inds]

    predictions['bboxes'] = bboxes
    predictions['scores'] = scores
    predictions['labels'] = labels

    print(predictions)

    visualize(os.path.join(result_dir_path, filename), img, predictions, colors=None, mask_display=False, class_names=class_names)