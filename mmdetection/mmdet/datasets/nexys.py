import mmcv
import numpy as np

from .custom import CustomDataset


class NexysDataset(CustomDataset):
    """
    Annotation format: 또한 "json", "yaml/yml" and "pickle/pkl" 파일 지원
    [
        {
            'filename': 'a.jpg', or 'img_prefix'을 제외한 path
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field),
                'mask_paths': mask paths (n, ) (with_mask=True, optional field)
            }
        },
        ...
    ]
    """
    CLASSES = ('pneumonia', )

    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]['ann']

        # 현재는 json 방식으로 저장하므로, list에서 numpy로 convert 해야 한다.
        gt_bboxes = ann_info['bboxes']
        gt_labels = ann_info['labels']
        gt_bboxes_ignore = ann_info['bboxes_ignore']

        if len(gt_bboxes) != 0:
            ann_info['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            ann_info['labels'] = np.array(gt_labels, dtype=np.int64)
        else:
            ann_info['bboxes'] = np.zeros((0, 4), dtype=np.float32)
            ann_info['labels'] = np.array([], dtype=np.int64)

        if len(gt_bboxes_ignore) != 0:
            ann_info['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
            ann_info['labels_ignore'] = np.array(ann_info['labels_ignore'], dtype=np.int64)
        else:
            ann_info['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
            ann_info['labels_ignore'] = np.array([], dtype=np.int64)

        if self.with_mask:
            mask_paths = ann_info['mask_paths']

            masks = list()
            for mask_path in mask_paths:
                mask = mmcv.imread(mask_path, flag='grayscale')
                masks.append(mask)

            ann_info['masks'] = masks

        return ann_info