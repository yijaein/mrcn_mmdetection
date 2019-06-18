import os
import json
import math
import mmcv
import time
import numpy as np

import csv
import pickle

from mmcv.runner import load_checkpoint

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from mmdetection.tools.visualize import visualize, random_colors
from mmdetection.mmdet.core import get_classes
from mmdetection.mmdet.models import build_detector
from mmdetection.mmdet.apis import inference_detector


config_path = '/home/bong3/lib/robin_mrcnn/mmdetection/configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_iitp.py'
# checkpoint_path = '/home/bong3/lib/robin_mrcnn/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_iitp/latest.pth'
checkpoint_path = "/home/bong3/lib/robin_mrcnn/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_iitp_1920_with_coco_vis_x4/epoch_5.pth"

cfg = mmcv.Config.fromfile(config_path)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, checkpoint_path)

# test a list of images
# list 폴더 형식
# img_dir_path = "/home/bong3/data/iitp/[트랙 1] 상황인지/t1_video"
# one_dir_checked = False
# img_dir_path = "/home/bong3/data/iitp/track1/my_samples_crop/t1_video_00054"
# one_dir_checked = True
# img_dir_path = "/home/bong3/data/iitp/track1/drone_sample/t1_video_00052"
# one_dir_checked = True
img_dir_path = "/home/bong3/data/iitp/track1/drone_sample"
one_dir_checked = False
result_dir_path = '/home/bong3/data/iitp/track1/sample_result'
gt_ann_path = '/home/bong3/data/iitp/track1/iitp_annotation.json'
result_info_save_path = "/home/bong3/data/iitp/track1/pkl/result_info_epoch_5_sample.pkl"
result_csv_path = os.path.join(result_dir_path, "result_sample.csv")
is_pkl_save = False
is_pkl_load = True
is_visualize = False


# 최대 bbox size 일때의 feature 기록 / 프레임 별 좌표 값 기록 / 화면 밖을 벗어났는지 확인(작업중)
class ObjectInfo(object):
    def __init__(self, label, feature, bbox, score, id):
        self.label = label
        self.feature = feature
        self.bbox = bbox
        self.score = score
        self.id = id
        self.max_bbox_feature = 0.0
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.bbox_by_frame = [bbox[:2]]
        self.feature_list = [feature]

    def update(self, feature, bbox, score):
        self.bbox = bbox
        self.score = score
        self.bbox_by_frame.append(bbox[:2])

        self.set_diff_feature(feature)
        self.feature = np.mean([self.feature, feature], axis=0)

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if self.area < area:
            self.max_bbox_feature = feature
            self.area = area

    def predict_bbox_update(self):
        predict_coordinate = self.get_predict_bbox()

        if len(predict_coordinate) > 0:
            self.bbox_by_frame.append(predict_coordinate)

    def get_predict_bbox(self):
        delta_coord_lsit = list()
        len_bbox = len(self.bbox_by_frame)
        for i in range(max(len_bbox-30, 0), len_bbox):
            delta_coord_lsit.append(np.subtract(self.bbox_by_frame[i], self.bbox_by_frame[i-1]))

        if len(delta_coord_lsit) > 0:
            mean_coord = np.mean(delta_coord_lsit, axis=0)
            return np.add(self.bbox_by_frame[-1], mean_coord)
        else:
            return delta_coord_lsit

    def set_diff_feature(self, feature):
        dist_feat = feature - self.feature
        dist_feat = np.dot(dist_feat, dist_feat)

        if dist_feat > 100:
            self.feature_list.append(feature)

if one_dir_checked:
    img_dirs = [img_dir_path]
else:
    img_dirs = [os.path.join(img_dir_path, filename) for filename in os.listdir(img_dir_path)]

dataset = 'iitp'
colors = random_colors(100)

if is_pkl_save:
    pkl_output = open(result_info_save_path, 'wb')
    result_info_dict = dict()

if is_pkl_load:
    pkl_file = open(result_info_save_path, 'rb')
    result_info_dict = pickle.load(pkl_file)
    pkl_file.close()

# CLASS = {"사람": 0, "소화기":1, "소화전":2, "차량":3, "자전거":4, "오토바이":5}
# 클래스 별 스코어 thr 설정
def checked_score(label):
    if label == 0:
        return 0.8
    elif label == 1:
        return 0.9
    elif label == 2:
        return 0.9
    elif label == 3:
        return 0.9
    elif label == 4:
        return 0.8
    elif label == 5:
        return 0.9

start_time = time.time()
for dir_path in sorted(img_dirs):
    imgs = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]
    imgs = sorted(imgs)

    dir_name = os.path.split(dir_path)[-1]
    dir_path = os.path.join(result_dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    predictions = dict()
    # CLASS = {"사람": 0, "소화기":1, "소화전":2, "차량":3, "자전거":4, "오토바이":5}
    objects_counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    # 일단은 무식하게 frame 별 object 들을 관리해본다. IITP 전용 300개
    frame_dict = [{0: list(), 1: list(), 2: list(), 3: list(), 4: list(), 5: list()} for _ in range(len(imgs))]

    def _process_result(i, result):
        # mask가 없을 때만 1번이다. mask 학습이었다면 2번이어야 한다... 지금은 귀찮음
        bb_result, img_feats = result

        class_names = get_classes(dataset)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bb_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bb_result)

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        bboxes = bboxes[:, :-1]

        # print(i, scores, labels)
        # if i == 23:
        #     exit()

        _scores = list()
        _bboxes = list()
        _labels = list()
        # CLASS = {"사람": 0, "소화기":1, "소화전":2, "차량":3, "자전거":4, "오토바이":5}
        if i == 0:
            for now_label_index, label in enumerate(labels):
                limit_score = checked_score(label)
                score = scores[now_label_index]
                if score < limit_score:
                    continue

                feature = img_feats[now_label_index]
                bbox = bboxes[now_label_index]

                # label, feature, bbox, score, id
                object_info = ObjectInfo(label, feature, bbox, score, objects_counter[label])
                frame_dict[i][label].append(object_info)
                objects_counter[label] += 1

                _scores.append(score)
                _labels.append(label)
                _bboxes.append(bbox)
        else:
            for now_label_index, label in enumerate(labels):
                # temp_dist = list()
                limit_score = checked_score(label)
                score = scores[now_label_index]
                if score < limit_score:
                    continue

                feature = img_feats[now_label_index]
                bbox = bboxes[now_label_index]

                _scores.append(score)
                _labels.append(label)
                _bboxes.append(bbox)

                # 같은 라벨에 대해서 전부다 하는 것이 아닌 bbox가 가까운 아이들만 검사한다.
                # 좌표 가져오기
                x1, y1, x2, y2 = bbox

                more_before_dist = list()
                # 사람, 소화기, 소화전, 자전거 // 자동차,오토바이 따로 구분해서 처리
                if label in [0,1,2,4]:
                    # 원래는 검사하는 프레임을 i-20으로 20프레임 전만 검사하지만 현재는 테스트 중이므로 50 프레임을 검사한다.
                    # max(i - 20, 0)
                    for before_frame_index in range(max(i - 50, 0), i):
                        more_before_objects = frame_dict[before_frame_index]

                        more_before_label_objects = more_before_objects[label]

                        # debug 용
                        # if len(more_before_label_objects)>0:
                        #     print("index:", before_frame_index, "label:", label)

                        for object_index, before_object in enumerate(more_before_label_objects):
                            # 엇나갈 좌표의 범위는 사람 기준 dist_frame*50.0 px로 둔다.
                            # CLASS = {"사람": 0, "소화기":1, "소화전":2, "차량":3, "자전거":4, "오토바이":5}
                            # 소화기 소화전은 드론 속도만을 고려 하여 프레임당 30px 이동으로 준다.
                            dist_frame = i - before_frame_index
                            if label == 0:
                                dist_px = min(650.0, dist_frame * 120.0)
                            elif label == 1:
                                dist_px = min(650.0, dist_frame * 120.0)
                            elif label == 2:
                                dist_px = min(650.0, dist_frame * 120.0)
                            elif label == 3:
                                dist_px = min(650.0, dist_frame * 200.0)
                            elif label == 4:
                                dist_px = min(650.0, dist_frame * 120.0)
                            else:
                                dist_px = min(650.0, dist_frame * 200.0)

                            b_x1, b_y1, b_x2, b_y2 = before_object.bbox

                            # a = x1 - b_x1  # 선 a의 길이
                            # b = y1 - b_y1  # 선 b의 길이

                            # print((x1, y1), "//", (b_x1, b_y1), "//", before_object.id)
                            # 테스트용 코드 작성
                            pred_coord_dist = 1000.0
                            predict_coordinate = before_object.get_predict_bbox()
                            if len(predict_coordinate) > 0:
                                _a = x1 - predict_coordinate[0]  # 선 a의 길이
                                _b = y1 - predict_coordinate[1]  # 선 b의 길이
                                pred_coord_dist = math.sqrt((_a * _a) + (_b * _b))

                            # bbox가 최대치 일때 비교와 이전 프레임의 feature 비교를 한다.
                            # 넓이는 최대 8*8 = 64 px 까지는 여유로 둔다.
                            max_dist_feat = 1000.0
                            if max((before_object.area - 64), 0) <= (x2 - x1) * (y2 - y1):
                                max_dist_feat = feature - before_object.max_bbox_feature
                                max_dist_feat = np.dot(max_dist_feat, max_dist_feat)

                            _dist_feat = feature - before_object.feature
                            _dist_feat = np.dot(_dist_feat, _dist_feat)

                            dist_feat_list = [_dist_feat]
                            for before_freature in before_object.feature_list:
                                _dist_feat = feature - before_freature
                                _dist_feat = np.dot(_dist_feat, _dist_feat)

                                dist_feat_list.append(_dist_feat)

                            dist_feat = min(dist_feat_list)

                            # 일단은 주위 100px 사이로만 검색 한다. (드론 이동이 있음) // 15 프레임까지는 거리 비교를 해본다.
                            # 또한 너무 비슷 하다면 기회를(?) 줘본다.
                            # print(dist_feat, max_dist_feat, dist_frame, "//", before_object.id, "//", pred_coord_dist)
                            dist_px = dist_px if dist_frame < 15 else 100
                            if min(x1+dist_px,1920) >= b_x1 and max(x1-dist_px,0) <= b_x1 and \
                                    min(y1 + dist_px, 1080) >= b_y1 and max(y1 - dist_px, 0) <= b_y1 or \
                                    (pred_coord_dist < 150 and dist_frame < 30):
                                # boundary 안에 있더라도 너무 안닮았으면 패스 한다.
                                if dist_feat < 200:
                                    more_before_dist.append([dist_feat*pred_coord_dist, before_frame_index, object_index])
                            elif dist_feat < 12:
                                more_before_dist.append([dist_feat*pred_coord_dist, before_frame_index, object_index])
                            elif max_dist_feat < 8:
                                more_before_dist.append([dist_feat*pred_coord_dist, before_frame_index, object_index])
                else:
                    # 원본 max(i - 5, 0) 자동차 15프레임이였음 (10프레임이 현재는 가장 스코어가 좋음)
                    for before_frame_index in range(max(i - 30, 0), i):
                        # print("index:", before_frame_index)
                        more_before_objects = frame_dict[before_frame_index]

                        more_before_label_objects = more_before_objects[label]
                        for object_index, before_object in enumerate(more_before_label_objects):
                            b_x1, b_y1 = before_object.bbox[:2]

                            # a = x1 - b_x1  # 선 a의 길이
                            # b = y1 - b_y1  # 선 b의 길이
                            # coord_dist = math.sqrt((a * a) + (b * b))

                            dist_frame = i - before_frame_index
                            if label == 3:
                                dist_px = min(650.0, dist_frame * 210.0)
                            else:
                                dist_px = min(650.0, dist_frame * 210.0)

                            # 테스트용 코드 작성
                            pred_coord_dist = 1000.0
                            predict_coordinate = before_object.get_predict_bbox()
                            if len(predict_coordinate) > 0:
                                _a = x1 - predict_coordinate[0]  # 선 a의 길이
                                _b = y1 - predict_coordinate[1]  # 선 b의 길이
                                pred_coord_dist = math.sqrt((_a * _a) + (_b * _b))

                            # bbox가 최대치 일때 비교와 이전 프레임의 feature 비교를 한다.
                            # 넓이는 최대 25*25 = 625 px 까지는 여유로 둔다.
                            max_dist_feat = 1000.0
                            if max((before_object.area - 625), 0) <= (x2 - x1) * (y2 - y1):
                                max_dist_feat = feature - before_object.max_bbox_feature
                                max_dist_feat = np.dot(max_dist_feat, max_dist_feat)

                            # dist_feat = feature - before_object.feature
                            # dist_feat = np.dot(dist_feat, dist_feat)
                            _dist_feat = feature - before_object.feature
                            _dist_feat = np.dot(_dist_feat, _dist_feat)

                            dist_feat_list = [_dist_feat]
                            for before_freature in before_object.feature_list:
                                _dist_feat = feature - before_freature
                                _dist_feat = np.dot(_dist_feat, _dist_feat)

                                dist_feat_list.append(_dist_feat)

                            dist_feat = min(dist_feat_list)

                            # 일단은 주위 100px 사이로만 검색 한다. (드론 이동이 있음) // 15 프레임까지는 거리 비교를 해본다.
                            # 또한 너무 비슷 하다면 기회를(?) 줘본다.
                            # print((x1, y1), "//", (b_x1, b_y1), "//", before_object.id, "//", dist_feat, max_dist_feat)
                            # print(dist_feat, max_dist_feat, dist_frame, "//", before_object.id, "//", pred_coord_dist)
                            dist_px = dist_px if dist_frame < 10 else 300
                            if min(x1 + dist_px, 1920) >= b_x1 and max(x1 - dist_px, 0) <= b_x1 and \
                                    min(y1 + dist_px, 1080) >= b_y1 and max(y1 - dist_px, 0) <= b_y1 or \
                                    (pred_coord_dist < 300 and dist_frame < 30):
                                # boundary 안에 있더라도 너무 안닮았으면 패스 한다.
                                if dist_feat < 180:
                                    more_before_dist.append([dist_feat * pred_coord_dist, before_frame_index, object_index])
                            elif dist_feat < 6:
                                more_before_dist.append(
                                    [dist_feat * pred_coord_dist, before_frame_index, object_index])
                            elif max_dist_feat < 3:
                                more_before_dist.append(
                                    [dist_feat * pred_coord_dist, before_frame_index, object_index])

                if len(more_before_dist) == 0:
                    # print("??", i)
                    objects_counter[label] += 1

                    # label, feature, bbox, score
                    object_info = ObjectInfo(label, feature, bbox, score, objects_counter[label])
                    frame_dict[i][label].append(object_info)
                else:
                    more_before_dist = np.array(more_before_dist)
                    min_index = np.argmin(more_before_dist[:, 0])
                    choice_frame = int(more_before_dist[min_index][1])
                    choice_index = int(more_before_dist[min_index][2])

                    more_before_objects = frame_dict[choice_frame][label]

                    similarity_object = more_before_objects[choice_index]
                    similarity_object.update(feature, bbox, score)

                    frame_dict[i][label].append(similarity_object)

                    del more_before_objects[choice_index]

        # 선택 받지 못한 예전 frame 들의 objects 들의 움직을 예측하여 업데이트 한다.
        # 다만 30 프레임을 벗어난 아이들은 갱신하지 않는다.
        for before_frame_index in range(max(i - 30, 0), i):
            more_before_objects = frame_dict[before_frame_index]
            more_before_label_objects = []
            for label_index in range(6):
                more_before_label_objects.extend(more_before_objects[label_index])

            for before_object in more_before_label_objects:
                before_object.predict_bbox_update()

        predictions['info'] = frame_dict[i]
        predictions['bboxes'] = _bboxes
        predictions['scores'] = _scores
        predictions['labels'] = _labels

        print(objects_counter)

        if is_visualize:
            visualize(os.path.join(dir_path, filename), mmcv.imread(file_path), predictions, colors=colors, mask_display=False, class_names=class_names)

    if is_pkl_load:
        for i in range(len(imgs)):
            print(i, imgs[i])

            file_path = imgs[i]
            filename = os.path.split(file_path)[-1]

            if file_path in result_info_dict:
                result = result_info_dict[file_path]
                _process_result(i, result)
    else:
        for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
            print(i, imgs[i])

            file_path = imgs[i]
            filename = os.path.split(file_path)[-1]

            if is_pkl_save:
                result_info_dict[file_path] = result
                continue

            _process_result(i, result)

    with open(result_csv_path, 'a') as wf:
        csv_wf = csv.writer(wf)
        csv_wf.writerow([objects_counter[0], objects_counter[1], objects_counter[2], objects_counter[3],
                         objects_counter[4], objects_counter[5]])

print("--- %s seconds ---" %(time.time() - start_time))

# result 저장용
if is_pkl_save:
    pickle.dump(result_info_dict, pkl_output)
    pkl_output.close()