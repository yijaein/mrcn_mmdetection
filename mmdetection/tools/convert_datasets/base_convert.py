import os
import json

from tqdm import tqdm
from collections import OrderedDict
from PIL import Image

"""
Annotation format: 또한 "json", "yaml/yml" and "pickle/pkl" 파일 지원
현재는 json만 지원하므로 np.ndarray 형식으로 저장 할 수 없다. (오직 list만 가능)
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


class BaseConvert(object):
    def __init__(self, root_path, out_path, mask_root_path=None, ann_path=None):
        self.root_path = root_path
        self.out_path = out_path
        self.mask_root_path = mask_root_path
        self.ann_path = ann_path

        self.ann_info = list()

        # {filename: idx...}
        self.file_index = dict()

    def load_img_info(self):
        forward_slash_check = self.root_path[-1] == '/'

        cnt = 0
        for dirName, subdirList, fileList in os.walk(self.root_path):
            for filename in tqdm(fileList, desc='file_progress'):
                if os.path.splitext(filename)[-1].lower() in [".png", ".jpg", ".jpeg"]:
                    file_path = os.path.join(dirName, filename)

                    img = Image.open(file_path)
                    width, height = img.size

                    if forward_slash_check:
                        exclude_root_path = file_path.replace(self.root_path, '')
                    else:
                        exclude_root_path = file_path.replace(self.root_path + '/', '')

                    file_info = OrderedDict()
                    file_info['filename'] = exclude_root_path
                    file_info['width'] = width
                    file_info['height'] = height

                    self.ann_info.append(file_info)
                    self.file_index[filename] = cnt

                    cnt += 1

    def load_ann_info(self):
        pass

    def load_mask_info(self):
        pass

    # json format
    def save_ann_file(self):
        with open(self.out_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.ann_info, json_file, ensure_ascii=False, indent='\t')

    def build(self):
        print("load images....")
        self.load_img_info()

        if self.mask_root_path is not None:
            print("load masks....")
            self.load_mask_info()

        if self.ann_path is not None:
            print("load annotations...")
            self.load_ann_info()
