import os
import csv
import cv2
import numpy as np

import torch
import torch.utils.data
from PIL import Image
from imgaug import augmenters as iaa

from facebook_mrcnn.maskrcnn_benchmark.structures.bounding_box import BoxList
from facebook_mrcnn.maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from facebook_mrcnn.maskrcnn_benchmark.data.augmentation.data_augmentation import img_and_key_point_augmentation


class RSNADataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "pneumonia"
    )

    def __init__(self, ann_file, root, remove_images_without_annotations, mask_type='polygon', transforms=None):
        # "mask_type" = "polygon" or "image"

        self.mask_type = mask_type
        self.img_key_list = list()
        self.img_dict = dict()
        self.ann_info = dict()

        cls = RSNADataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        for dirName, subdirList, fileList in os.walk(root):
            for filename in fileList:
                filename, ext = os.path.splitext(filename)
                if ext.lower() in [".png", ".jpg", ".jpeg"]:
                    self.img_dict[filename] = os.path.join(dirName, filename + ext)
                    self.ann_info[filename] = list()

        # csv 용도 이며, mask 이미지인 경우 다르게 작업
        with open(ann_file, 'r') as ann_f:
            ann_cvf = csv.reader(ann_f)

            # patientId,x,y,width,height,Target
            for i, line in enumerate(ann_cvf):
                if i == 0:
                    continue

                filename, x, y, w, h, target = line
                target = int(target)

                if remove_images_without_annotations:
                    if target == 0:
                        continue

                    x1 = int(x)
                    y1 = int(y)
                    w = int(w)
                    h = int(h)

                    x2 = x1 + w
                    y2 = y1 + h

                    self.img_key_list.append(filename)
                else:
                    self.img_key_list.append(filename)
                    x1 = 0
                    y1 = 0
                    x2 = 0
                    y2 = 0

                try:
                    self.ann_info[filename].append([x1,y1,x2,y2,target])
                except KeyError:
                    continue

        self.augmentation = iaa.Sequential([
            iaa.OneOf([  ## geometric transform
                iaa.Affine(
                    scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
                    rotate=(-2, 2),
                    shear=(-1, 1),
                ),
                iaa.PiecewiseAffine(scale=(0.001, 0.025)),
            ]),
            iaa.OneOf([  ## brightness or contrast
                iaa.Multiply((0.9, 1.1)),
                iaa.ContrastNormalization((0.9, 1.1)),
            ]),
            iaa.OneOf([  ## blur or sharpen
                iaa.GaussianBlur(sigma=(0.0, 0.1)),
                iaa.Sharpen(alpha=(0.0, 0.1)),
            ]),
        ])

        # 중복 방지 RSNA csv 파일 참조
        self.img_key_list = list(set(self.img_key_list))
        self.transforms = transforms

    def __getitem__(self, idx):
        filename = self.img_key_list[idx]

        img = cv2.imread(self.img_dict[filename], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = Image.open(self.img_dict[filename]).convert("RGB")
        height, width = img.shape[:2]

        # target settings
        boxes = []
        temp_masks = []
        masks = []
        gt_classes = []

        for ann_info in self.ann_info[filename]:
            mask = np.zeros((height, width))

            bndbox = ann_info[:4]
            mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1

            x1, y1, x2, y2 = ann_info[:4]
            temp_mask = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            temp_masks.append(temp_mask)

            boxes.append(bndbox)
            masks.append([mask])

            # 만약 클래스가 번호가 아닌 이름으로 있다면 아래 코드를 사용한다.
            # gt_classes.append(self.class_to_ind[ann_info[-1]])
            gt_classes.append(ann_info[-1])

        img, boxes, temp_masks= img_and_key_point_augmentation(self.augmentation, img, boxes, temp_masks)
        img = Image.fromarray(img, mode="RGB")

        target = BoxList(torch.tensor(boxes, dtype=torch.float32), (width, height), mode="xyxy")
        target.add_field("labels", torch.tensor(gt_classes))

        seg_masks = SegmentationMask(temp_masks, (width, height), type=self.mask_type)
        target.add_field("masks", seg_masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.img_key_list)

    def _get_image_polygons(self, mask):
        _, contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        polygon = list()
        box_list = list()
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x1, y1, w, h = rect
            box = [x1, y1, x1 + w, y1 + h]

            contour = np.reshape(contour, (contour.shape[0], contour.shape[2]))
            # contour = np.fliplr(contour)
            contour = np.reshape(contour, (contour.shape[0] * contour.shape[1])).tolist()

            polygon.append([contour])
            box_list.append(box)

        return polygon, box_list

    def get_img_info(self, index):
        return {"height": 512, "width": 512}

    def map_class_id_to_class_name(self, class_id):
        return RSNADataset.CLASSES[class_id]
