import os
import csv

from PIL import Image
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


class VisDroneConvert(BaseConvert):

    def load_img_info(self):
        cnt = 0
        for dir_path, subdir_list, file_list in os.walk(self.root_path):
            for filename in tqdm(file_list, desc='file_progress'):
                file_path = os.path.join(dir_path, filename)

                img = Image.open(file_path)
                width, height = img.size

                dir_name = os.path.basename(dir_path)
                change_filename = dir_name + "_" + filename

                file_info = OrderedDict()
                file_info['filename'] = change_filename
                file_info['width'] = width
                file_info['height'] = height

                self.ann_info.append(file_info)

                img_num = int(os.path.splitext(filename)[0])
                if dir_name not in self.file_index:
                    self.file_index[dir_name] = [(cnt, img_num)]
                else:
                    self.file_index[dir_name].append((cnt, img_num))

                cnt += 1

                # img.save(os.path.join("/home/bong3/data/iitp/Vis Drone/찜/img_copy", change_filename))

    def load_ann_info(self):
        for filename in os.listdir(self.ann_path):
            ann_txt_path = os.path.join(self.ann_path, filename)

            anno_info_dict = dict()
            with open(ann_txt_path, 'r') as f:
                csv_f = csv.reader(f, delimiter=',')

                for row in csv_f:
                    sequence_index = int(row[0])
                    category_id = int(row[7])
                    x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])

                    if sequence_index not in anno_info_dict:
                        # 사람
                        if category_id == 1 or category_id == 2:
                            anno_info_dict[sequence_index] = [(1, x, y, x+w, y+h)]
                        # 차량
                        elif category_id == 4 or category_id == 5 or category_id == 6 or category_id == 9:
                            anno_info_dict[sequence_index] = [(4, x, y, x + w, y + h)]
                        # 자전거
                        elif category_id == 3:
                            anno_info_dict[sequence_index] = [(5, x, y, x + w, y + h)]
                        # 오토바이
                        elif category_id == 10:
                            anno_info_dict[sequence_index] = [(6, x, y, x + w, y + h)]
                    else:
                        # The object category indicates the type of annotated object, (i.e., ignored regions (0), pedestrian (1),
                        #                       people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10),
                        # 	              others (11))
                        # 사람
                        if category_id == 1 or category_id == 2:
                            anno_info_dict[sequence_index].append((1, x, y, x + w, y + h))
                        # 차량
                        elif category_id == 4 or category_id == 5 or category_id == 6 or category_id == 9:
                            anno_info_dict[sequence_index].append((4, x, y, x + w, y + h))
                        # 자전거
                        elif category_id == 3:
                            anno_info_dict[sequence_index].append((5, x, y, x + w, y + h))
                        # 오토바이
                        elif category_id == 10:
                            anno_info_dict[sequence_index].append((6, x, y, x + w, y + h))

            dir_name = os.path.splitext(filename)[0]
            for cnt, img_num in self.file_index[dir_name]:
                file_info = self.ann_info[cnt]
                # print(file_info)

                if 'ann' not in file_info:
                    ann_info = file_info['ann'] = OrderedDict()
                else:
                    ann_info = file_info['ann']

                if int(img_num) not in anno_info_dict:
                    ann_info['bboxes'] = []
                    ann_info['labels'] = []
                    ann_info['bboxes_ignore'] = []
                    ann_info['labels_ignore'] = []
                else:
                    bboxes = list()
                    labels = list()
                    for label, x1, y1, x2, y2 in anno_info_dict[int(img_num)]:
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(label)

                    ann_info['bboxes'] = bboxes
                    ann_info['labels'] = labels
                    ann_info['bboxes_ignore'] = []
                    ann_info['labels_ignore'] = []