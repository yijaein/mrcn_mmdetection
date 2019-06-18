import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import os
import cv2

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from facebook_mrcnn.maskrcnn_benchmark.config import cfg
from facebook_mrcnn.demo.predictor_coco import COCODemo

# config_file ="../configs/e2e_mask_rcnn_SE_50_FPN_1x_rsna_test.yaml"
# config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x_rsna_test.yaml"
config_file = "../configs/e2e_mask_rcnn_R_101_FPN_1x_coco_test.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

# coco_demo = RSNADemo(
#     cfg,
#     min_image_size=512,
#     confidence_threshold=0.6,
# )

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.1,
)

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

# dir_path = "/home/bong9/data/rsna/rsna512/test"
dir_path = "/media/bong9/3EACEB5D71A84DBB/coco/test"
anno_file = "/home/bong9/data/rsna/rsna512/test_labels_512.csv"

img_list = os.listdir(dir_path)

# anno_dict = dict()
# for filename in img_list:
#     filename = os.path.splitext(filename)[0]
#     anno_dict[filename] = list()
#
# with open(anno_file, 'r') as ann_f:
#     ann_cvf = csv.reader(ann_f)
#
#     # patientId,x,y,width,height,Target
#     for i, line in enumerate(ann_cvf):
#         if i == 0:
#             continue
#
#         filename, x, y, w, h, target = line
#         target = int(target)
#
#         if target == 0:
#             continue
#
#         x1 = int(x)
#         y1 = int(y)
#         w = int(w)
#         h = int(h)
#
#         x2 = x1 + w
#         y2 = y1 + h
#
#         anno_dict[filename].append((x1, y1, x2, y2))


for filename in img_list:
    file_path = os.path.join(dir_path, filename)

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    predictions = coco_demo.run_on_opencv_image(img)

    # filename = os.path.splitext(filename)[0]
    # anno_info_list = anno_dict[filename]
    #
    # for anno_info in anno_info_list:
    #     cv2.rectangle(predictions, anno_info[:2], anno_info[2:4], (255,0,0), 2)

    imshow(predictions)


