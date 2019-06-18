import os

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from imgaug import augmenters as iaa

from facebook_mrcnn.maskrcnn_benchmark.data.augmentation.data_augmentation import img_and_key_point_augmentation
from facebook_mrcnn.maskrcnn_benchmark.structures.bounding_box import BoxList
from facebook_mrcnn.maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


# 이 파일과 같은 새로운 데이터로더 클래스 추가후에는
# ~/lib/robin_mrcnn/maskrcnn_benchmark/data/datasets/__init__.py 파일에 등록을 해야만
# ~/lib/robin_mrcnn/maskrcnn_benchmark/config/paths_catalog.py 에 지정한 Factory로 참조가 가능해짐

class LiverBgDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "liver",
        "bg"
    )

    def __init__(self, mask_dir=None, root=None, mask_type=None, transforms=None, is_train=True):
        print('Dataset Init: \n\tmask_dir={}, \n\troot={}, \n\tmask_type={}, \n\ttransforms={}, \n\tis_train={}'.format(
            mask_dir, root, mask_type, transforms, is_train))
        # "mask_type" = "polygon" or "image"

        # for debug
        print('data loader init args')
        init_list = [mask_dir, root, mask_type, transforms, is_train]
        print('\t' + '\n\t'.join([str(arg) if arg else 'None' for arg in init_list]))
        # end

        # norm path
        root = norm_path(root)
        mask_dir = norm_path(mask_dir)

        self.mask_type = 'polygon'  # mask_type
        self.transforms = transforms
        self.image_size = 512
        self.is_train = is_train
        self.img_key_list = list()
        self.img_dict = dict()
        self.ann_info = dict()

        cls = LiverBgDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        self.img_dict = image_dict(root, exts=['.png', '.jpg', '.jpeg'], recursive=True, followlinks=True)

        for cls_num, cls_name in enumerate(self.CLASSES[1:], 1):
            cls_mask_path = os.path.join(mask_dir, cls_name)
            assert os.path.exists(cls_mask_path), "Not found class({}) path: {}".format(cls, cls_mask_path)

            mask_dict = image_dict(cls_mask_path)
            for mask_key, mask_file in mask_dict.items():
                if mask_key not in self.ann_info:
                    self.ann_info[mask_key] = list()
                # x, y, w, h = None, None, None, None
                # self.ann_info[mask_key].append([x, y, w, h, cls_num, mask_file])
                self.ann_info[mask_key].append([cls_num, mask_file])

        self.img_key_list = list(set(self.img_dict) & set(self.ann_info))
        print('found images', len(self.img_dict))
        print('found masks', len(self.ann_info))
        print('using image&mask', len(self.img_key_list))

        # self.train_augmentation = iaa.Sequential([
        #     # iaa.PiecewiseAffine(scale=(0.00, 0.05), nb_cols=3, nb_rows=3),
        #     # iaa.Affine(rotate=(-20, 20)),
        #     iaa.SomeOf((0, None), [
        #         # iaa.Fliplr(0.5),
        #         iaa.Multiply((0.5, 1.5)),
        #         iaa.Add((-10, 10)),
        #         iaa.GaussianBlur(sigma=(0, 1.0))
        #     ], random_order=False)
        # ])

        if self.is_train:
            self.augmentation = iaa.Sequential([
                iaa.SomeOf((0, None), [
                    iaa.Fliplr(0.5),
                    iaa.PiecewiseAffine(scale=(0.00, 0.05), nb_cols=3, nb_rows=3),
                    iaa.Affine(rotate=(-20, 20)),
                    iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)}),
                    iaa.Affine(scale=(0.5, 1.5)),
                    iaa.Multiply((0.5, 1.5)),
                    iaa.Add((-10, 10)),
                    iaa.GaussianBlur(sigma=(0, 1.0))
                ], random_order=False)
            ])
        else:
            self.augmentation = iaa.Sequential([], random_order=False)

    def __getitem__(self, idx):
        filename = self.img_key_list[idx]

        while True:
            img_aug, target = self.get_groundtruth(filename)
            target = target.clip_to_image(remove_empty=True)

            if len(target) > 0:
                break
            else:
                print('Warning: zero of target boxes', filename)
                continue

        img_aug = Image.fromarray(img_aug, mode="RGB")

        if self.transforms is not None:
            img_aug, target = self.transforms(img_aug, target)

        return img_aug, target, idx

    def __len__(self):
        return len(self.img_key_list)

    def get_groundtruth(self, filename):
        img = cv2.imread(self.img_dict[filename], cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_keep_aspect_ratio_with_padding(img)
        height, width = img.shape[:2]

        boxes = []
        masks = []
        gt_classes = []

        for ann_info in self.ann_info[filename]:
            cls_num, mask_file = ann_info
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = self.resize_keep_aspect_ratio_with_padding(mask)

            # 하나의 클래스의 여러 분절된 세그먼테이션을 모두 포함하는 바운딩 박스를 구함
            bbox_points, mask_points = self.find_bounding_square(mask)

            x1, y1, x2, y2 = self.getBiggestBoundbox(bbox_points)
            bbox = [x1, y1, x2, y2]
            # print('w:{} h:{} len:{} pts:{} x1<x2:{} y1<y2:{} name:{}'.format(x2-x1, y2-y1, len(bbox_points), bbox_points, x1<x2, y1<y2, filename))

            boxes.append(bbox)
            masks.append(mask_points)
            gt_classes.append(cls_num)

        img, boxes, masks = img_and_key_point_augmentation(self.augmentation, img, boxes, masks)

        anno = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "masks": masks,
            "labels": torch.tensor(gt_classes),
        }

        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        masks = SegmentationMask(anno["masks"], (width, height), type=self.mask_type)
        target.add_field("masks", masks)

        return img, target

    def getBiggestBoundbox(self, bbox_list):
        Xmin = min([xyxy[0] for xyxy in bbox_list])
        Ymin = min([xyxy[1] for xyxy in bbox_list])
        Xmax = max([xyxy[2] for xyxy in bbox_list])
        Ymax = max([xyxy[3] for xyxy in bbox_list])
        # x, y, w, h = Xmin, Ymin, Xmax-Xmin, Ymax-Ymin
        # return x, y, w, h
        x1, y1, x2, y2 = Xmin, Ymin, Xmax, Ymax
        return x1, y1, x2, y2

    def getBBox(self, cont):
        x, y, w, h = cv2.boundingRect(cont)
        x1, y1, x2, y2 = x, y, x + w, y + h
        return x1, y1, x2, y2

    def getBiggestBoundbox(self, xywhList):
        Xmin = min([xywh[0] for xywh in xywhList])
        Ymin = min([xywh[1] for xywh in xywhList])
        Wsum = max([xywh[0] + xywh[2] for xywh in xywhList]) - Xmin
        Hsum = max([xywh[1] + xywh[3] for xywh in xywhList]) - Ymin
        x, y, w, h = Xmin, Ymin, Wsum, Hsum
        return x, y, w, h

    def find_bounding_square(self, mask):
        mask = mask.astype(np.uint8)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        bboxs = [self.getBBox(contour) for contour in contours]

        mask_points = []
        for cont in contours:
            cont = np.array(cont)
            cont = cont.reshape((-1))
            # print(cont.shape)
            mask_points.append(cont.tolist())

        return bboxs, mask_points

    def resize_keep_aspect_ratio_with_padding(self, im):
        size = self.image_size

        old_size = im.shape[:2]  # old_size is in (height, width) format
        ratio = float(size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im

    # def get_image_polygons(self, mask):
    #     _, contours, hierarchy = cv2.findContours(
    #         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #     return 0
    #
    # def get_img_info(self, index):
    #     return {"height": 512, "width": 512}
    #
    # def map_class_id_to_class_name(self, class_id):
    #     return KidneyDataset.CLASSES[class_id]


# util functions


def norm_path(path, makedirs=False):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if makedirs and not os.path.exists(path):
        os.makedirs(path)
        print('makedirs:, path')
    return path


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


def test_dataset(num_worker=8, split=True):
    import multiprocessing

    dataset = LiverBgDataset(
        mask_dir='/home/bong07/data/yonsei2/dataset/SegMrcnn_kidney_liver_some_400_BG_20190404/seg',
        root='/home/bong07/data/yonsei2/dataset/SegMrcnn_kidney_liver_some_400_BG_20190404/us/train',
        mask_type='image',
        transforms=None,
        is_train=True)

    cnt_data = len(dataset)
    print('cnt of dataset: {}'.format(cnt_data))

    def job(start=0, end=None, sleep_time=0.5):
        continue_idx = 0
        idx = start

        if not end:
            end = len(dataset)

        while True:
            # start_time = time.time()
            _ = dataset[idx]
            # end_time = time.time()
            # print('{}\ttime:{:0.4f}, index:{}'.format(multiprocessing.current_process(), end_time - start_time, idx))
            # time.sleep(sleep_time)

            idx = idx + 1 if idx + 1 < end else start
            continue_idx += 1

    if num_worker > 1:
        # use multi-process
        childProcess_list = list()
        for idx in range(num_worker):
            if split:
                valume = cnt_data // num_worker
                start = idx * valume
                end = start + valume
            childProcess = multiprocessing.Process(target=job, args=(start, end))
            childProcess_list.append(childProcess)
            childProcess.start()
        for childProcess in childProcess_list:
            childProcess.join()
    else:
        # use main-process
        job()


if __name__ == '__main__':
    test_dataset(num_worker=1)
