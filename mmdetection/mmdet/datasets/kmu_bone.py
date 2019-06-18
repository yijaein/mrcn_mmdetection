import mmcv
import numpy as np
import cv2

from .custom import CustomDataset

import os
import os.path as osp
from mmcv.parallel import DataContainer as DC
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation


class KmuDataset(CustomDataset):
    """
    Annotation format: 또한 "json", "yaml/yml" and "pickle/pkl" 파일 지원
    [
        {
            'filename': 'a.jpg', or 'img_prefix'을 제외한 path
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <list> (n, 4) (x1,y1,x2,y2),
                'labels': <list> (n, ),
                'keypoints' : <list> (n, keypoints list) [x1,y1, x2,y2, ...]
                'bboxes_ignore': <list> (k, 4) (x1,y1,x2,y2),
                'labels_ignore': <list> (k, 4) (optional field),
                'keypoints_ignore' : <list> (k, keypoints list) [x1,y1, x2,y2, ...]
                'mask_paths': mask paths (n, ) (with_mask=True, optional field)
            }
        },
        ...
    ]
    """

    CLASSES = ('A1 Right', 'A1 Left', 'A2 Right', 'A2 Left', 'A3 Right', 'A3 Left',
               'B1 Right', 'B1 Left', 'C Right', 'C Left')


    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]['ann']
        image_width = self.img_infos[idx]['width']
        image_height = self.img_infos[idx]['height']

        # 현재는 json 방식으로 저장하므로, list에서 numpy로 convert 해야 한다.

        bboxes = ann_info['bboxes']
        labels = ann_info['labels']
        bboxes_ignore = ann_info['bboxes_ignore']
        labels_ignore = ann_info['labels_ignore']

        keypoints = ann_info['keypoints']
        # keypoints_ignore = ann_info['keypoints_ignore']

        if not bboxes:
            ann_info['bboxes'] = np.zeros((0, 4), dtype=np.float32)
            ann_info['labels'] = np.array([], dtype=np.int64)
        else:
            ann_info['bboxes'] = np.array(bboxes, dtype=np.float32)
            ann_info['labels'] = np.array(labels, dtype=np.int64)

        if not bboxes_ignore:
            ann_info['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
            ann_info['labels_ignore'] = np.array(labels, dtype=np.int64)
        else:
            ann_info['bboxes_ignore'] = np.array(bboxes_ignore, dtype=np.float32)
            ann_info['labels_ignore'] = np.array(labels_ignore, dtype=np.int64)

        # print(bboxes_ignore)
        # print('bboxes', ann_info['bboxes'])

        # if not keypoints:
        #     ann_info['keypoints'] = np.zeros((0, 2), dtype=np.float32)
        # else:
        #     print(keypoints)
        #     ann_info['keypoints'] = np.array(keypoints, dtype=np.float32)
        #
        # if not keypoints:
        #     ann_info['keypoints_ignore'] = np.zeros((0, 2), dtype=np.float32)
        # else:
        #     ann_info['keypoints_ignore'] = np.array(keypoints_ignore, dtype=np.float32)


        if self.with_mask:
            gt_masks = []

            for i in range(len(keypoints)):
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                class_id = labels[i]
                points = keypoints[i]

                x1, y1, x2, y2 = bboxes[i]
                mask[y1:y2,x1:x2] = 1

                circle_size = 9

                points_len = len(points) // 2

                points = np.array(points)
                pts = points.reshape((-1, 2))

                pts = sorted(pts, key=lambda x: (x[0]))

                for k in range(points_len):
                    px = pts[k][0]
                    py = pts[k][1]

                    if class_id == 3 or class_id == 4:
                        circle_size = 10

                    if class_id == 5 or class_id == 6 :
                        circle_size = 8

                    if class_id == 7 and k== 2:
                        circle_size = 10

                    if class_id == 9 and class_id == 10:
                        circle_size = 5

                    cv2.circle(mask, (px, py), circle_size, 1, -1)

                gt_masks.append(mask.astype(np.uint8))

            ann_info['masks'] = gt_masks

        return ann_info


