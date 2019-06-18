import shutil
import os
import json

from tqdm import tqdm
from pycocotools.coco import COCO
from collections import OrderedDict
# from .base_convert import BaseConvert


"""
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
"""

# 사람<0> 소화전<10> 차량<2,5,7> 자전거<1> 오토바이<3>
# class COCOConvert(BaseConvert):
#
#     def load_ann_info(self):
#         self.coco = COCO(self.ann_path)
#         self.cat_ids = self.coco.getCatIds()
#         self.cat2label = {
#             cat_id: i + 1
#             for i, cat_id in enumerate(self.cat_ids)
#         }
#         self.img_ids = self.coco.getImgIds()
#         img_infos = []
#         for i in self.img_ids:
#             info = self.coco.loadImgs([i])[0]
#             info['filename'] = info['file_name']
#             img_infos.append(info)

        # with open(self.ann_path, 'r') as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #
        #     for i, line in tqdm(enumerate(csv_reader), desc='load ann'):
        #         if i == 0:
        #             continue
        #
        #         # patientId,x,y,width,height,Target
        #         patient_id, x, y, width, height, target = line
        #
        #         filename = '%s.png'% patient_id
        #
        #         idx = self.file_index[filename]
        #         file_info = self.ann_info[idx]
        #
        #         if 'ann' not in file_info:
        #             ann_info = file_info['ann'] = OrderedDict()
        #         else:
        #             ann_info = file_info['ann']
        #
        #         target = int(target)
        #         if target == 0:
        #             ann_info['bboxes'] = []
        #             ann_info['labels'] = []
        #             ann_info['bboxes_ignore'] = []
        #             ann_info['labels_ignore'] = []
        #         else:
        #             x1 = int(x)
        #             y1 = int(y)
        #             width = int(width)
        #             height = int(height)
        #
        #             x2 = x1 + width
        #             y2 = y1 + height
        #
        #             bboxes = [[x1, y1, x2, y2]]
        #             labels = [target]
        #
        #             if 'bboxes' not in ann_info:
        #                 ann_info['bboxes'] = bboxes
        #             else:
        #                 # print(file_info['filename'], ann_info['bboxes'].shape, bboxes.shape)
        #                 ann_info['bboxes'].extend(bboxes)
        #
        #             if 'labels' not in ann_info:
        #                 ann_info['labels'] = labels
        #             else:
        #                 ann_info['labels'].extend(labels)
        #
        #             if 'bboxes_ignore' not in ann_info:
        #                 ann_info['bboxes_ignore'] =[]
        #
        #             if 'labels_ignore' not in ann_info:
        #                 ann_info['labels_ignore'] = []

# 걍 convert 안쓰고 따로 json으로 만듬
if __name__ == '__main__':
    img_dir_path = "/media/bong3/DATA/data/coco/train2014"
    result_img_path = "/home/bong3/data/iitp/track1/coco_train"
    coco = COCO("/media/bong3/DATA/data/coco/annotations/instances_train2014.json")
    cat_ids = coco.getCatIds()
    cat2label = {
        cat_id: i + 1
        for i, cat_id in enumerate(cat_ids)
    }
    img_ids = coco.getImgIds()
    img_infos = []
    for count, i in tqdm(enumerate(img_ids)):
        info = coco.loadImgs([i])[0]

        file_info = OrderedDict()
        file_info['filename'] = info['file_name']
        file_info['width'] = info['width']
        file_info['height'] = info['height']

        img_id = info['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)

        write_ann_info = file_info['ann'] = OrderedDict()

        # 사람<1> 소화전<11> 차량<3,6,8> 자전거<2> 오토바이<4>
        write_ann_info['labels'] = list()
        write_ann_info['bboxes'] = list()
        write_ann_info['bboxes_ignore'] = list()
        write_ann_info['labels_ignore'] = list()
        for annotation in ann_info:
            if annotation['iscrowd']:
                continue

            category_id = annotation['category_id']

            if category_id == 1 or category_id == 2 or category_id == 3 or \
                    category_id == 4 or category_id == 6 or category_id == 8: # or category_id == 11:

                if category_id == 1:
                    label = 1
                # elif category_id == 11:
                #     label = 3
                elif category_id == 3 or category_id == 6 or category_id == 8:
                    label = 4
                elif category_id == 2:
                    label = 5
                elif category_id == 4:
                    label = 6

                x1, y1, w, h = annotation['bbox']
                x2 = x1+w-1
                y2 = y1+h-1

                write_ann_info['labels'].append(label)
                write_ann_info['bboxes'].append([x1,y1,x2,y2])

        if len(write_ann_info['labels']) == 0:
            continue

        # shutil.copy(os.path.join(img_dir_path, info['file_name']), os.path.join(result_img_path, info['file_name']))
        img_infos.append(file_info)

    # # print(img_infos)
    with open("/home/bong3/data/iitp/track1/coco_annotation.json", 'w', encoding='utf-8') as json_file:
        json.dump(img_infos, json_file, ensure_ascii=False, indent='\t')