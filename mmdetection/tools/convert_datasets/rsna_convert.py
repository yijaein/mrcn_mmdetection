import csv
import numpy as np

from tqdm import tqdm
from collections import OrderedDict
from .base_convert import BaseConvert

"""
Annotation format: 또한 "json", "yaml/yml" and "pickle/pkl" 파일 지원
현재는 json만 지원하므로 np.ndarray 형식으로 저장 할 수 없다. (오직 list만 가능)
[
    {
        'filename': 'a.jpg', or 'img_prefix'을 제외한 path
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <list> (n, 4) (x1,y1,x2,y2),
            'labels': <list> (n, ),
            'bboxes_ignore': <list> (k, 4) (x1,y1,x2,y2),
            'labels_ignore': <list> (k, 4) (optional field),
            'mask_paths': mask paths (n, ) (with_mask=True, optional field)
        }
    },
    ...
]
"""

class RSNAConvert(BaseConvert):

    def load_ann_info(self):
        with open(self.ann_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for i, line in tqdm(enumerate(csv_reader), desc='load ann'):
                if i == 0:
                    continue

                # patientId,x,y,width,height,Target
                patient_id, x, y, width, height, target = line

                filename = '%s.png'% patient_id

                idx = self.file_index[filename]
                file_info = self.ann_info[idx]

                if 'ann' not in file_info:
                    ann_info = file_info['ann'] = OrderedDict()
                else:
                    ann_info = file_info['ann']

                target = int(target)
                if target == 0:
                    ann_info['bboxes'] = []
                    ann_info['labels'] = []
                    ann_info['bboxes_ignore'] = []
                    ann_info['labels_ignore'] = []
                else:
                    x1 = int(x)
                    y1 = int(y)
                    width = int(width)
                    height = int(height)

                    x2 = x1 + width
                    y2 = y1 + height

                    bboxes = [[x1, y1, x2, y2]]
                    labels = [target]

                    if 'bboxes' not in ann_info:
                        ann_info['bboxes'] = bboxes
                    else:
                        # print(file_info['filename'], ann_info['bboxes'].shape, bboxes.shape)
                        ann_info['bboxes'].extend(bboxes)

                    if 'labels' not in ann_info:
                        ann_info['labels'] = labels
                    else:
                        ann_info['labels'].extend(labels)

                    if 'bboxes_ignore' not in ann_info:
                        ann_info['bboxes_ignore'] =[]

                    if 'labels_ignore' not in ann_info:
                        ann_info['labels_ignore'] = []