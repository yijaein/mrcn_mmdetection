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
checkpoint_path = "/home/bong3/lib/robin_mrcnn/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_iitp_1920_with_coco_vis/epoch_5.pth"
# checkpoint_path = "/home/bong3/lib/robin_mrcnn/work_dirs/190526_epoch_12.pth"
# checkpoint_path = "/home/bong3/lib/robin_mrcnn/work_dirs/latest.pth"

cfg = mmcv.Config.fromfile(config_path)
cfg.model.pretrained = None

# test a list of images
# list 폴더 형식
# img_dir_path = "/home/bong3/data/iitp/[트랙 1] 상황인지/t1_video"
# one_dir_checked = False
# img_dir_path = "/home/bong3/data/iitp/[트랙 1] 상황인지/t1_video/t1_video_00001"
# one_dir_checked = True
img_dir_path = "/home/bong3/data/iitp/track1/drone_sample/t1_video_00070"
one_dir_checked = True
result_dir_path = '/home/bong3/data/iitp/track1/result'
gt_ann_path = '/home/bong3/data/iitp/track1/iitp_annotation.json'
# result_info_save_path = "/home/bong3/data/iitp/track1/pkl/result_info_epoch_20.pkl"
result_info_save_path = "/home/bong3/data/iitp/track1/pkl/result_info_190531_epoch_5_test70clip.pkl"
result_csv_path = os.path.join(result_dir_path, "result.csv")
is_pkl_save = False
is_pkl_load = True
is_visualize = False

# construct the model and load checkpoint
if not is_pkl_load:
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, checkpoint_path)
img_size = (1920, 1080)


# 최대 bbox size 일때의 feature 기록 / 프레임 별 좌표 값 기록 / 화면 밖을 벗어났는지 확인(작업중)
class ObjectInfo(object):
    def __init__(self, label, feature, bbox, score, id):
        self.label = label
        self.feature = feature
        self.bbox = bbox
        self.score = score
        self.id = id
        self.max_bbox_feature = 0.0
        # w, h
        self.max_bbox_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # bbox 센터 점을 넣어본다.
        # self.bbox_by_frame = [bbox[:2]]
        x1,y1,x2,y2 = bbox
        self.bbox_center_by_frame = [[(x2 + x1) / 2, (y2 + y1) / 2]]
        self.edge_dist = 5
        # 마지막 프레임의 벡터 방향을 기록한다. (0: left, 1: right, 2: up, 3: down)
        # self.last_direction = 0

        self.feature_list = [feature]

    def update(self, feature, bbox, score):
        self.bbox = bbox
        self.score = score

        self.set_diff_feature(feature)
        self.feature = np.mean([self.feature, feature], axis=0)

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        max_area = self.max_bbox_size[0]*self.max_bbox_size[1]
        if max_area < area:
            self.max_bbox_feature = feature
            self.max_bbox_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        self.area = area

        # bbox 센터 점을 넣어본다.
        # 각 면의 끝을 지나가는 아이들은 예측 값으로 간다.
        # self.bbox_by_frame.append(bbox[:2])
        x1, y1, x2, y2 = bbox
        if (x1 < self.edge_dist or y1 < self.edge_dist or \
            x2 > img_size[0]-self.edge_dist or y2 > img_size[1]-self.edge_dist) and \
                ((x2-x1) < self.max_bbox_size[0] or (y2-y1) < self.max_bbox_size[1]):

            half_w = self.max_bbox_size[0]/2
            half_h = self.max_bbox_size[1]/2

            if x1 < self.edge_dist and y1 < self.edge_dist:
                self.bbox_center_by_frame.append([x2 - half_w, y2 - half_h])
            elif x1 < self.edge_dist:
                self.bbox_center_by_frame.append([x2 - half_w, (y2 + y1) / 2])
            elif y1 < self.edge_dist:
                self.bbox_center_by_frame.append([(x2 + x1) / 2, y2 - half_h])
            elif x2 > img_size[0]-self.edge_dist and y2 > img_size[1]-self.edge_dist:
                self.bbox_center_by_frame.append([x1 + half_w, y1 + half_h])
            elif x2 > img_size[0]-self.edge_dist:
                self.bbox_center_by_frame.append([x1 + half_w, (y2 + y1) / 2])
            elif y2 > img_size[1]-self.edge_dist:
                self.bbox_center_by_frame.append([(x2 + x1) / 2, y1 + half_h])
        else:
            self.bbox_center_by_frame.append([(x2 + x1) / 2, (y2 + y1) / 2])

        # now_x, now_y = self.bbox_center_by_frame[-1]
        # before_x, before_y = self.bbox_center_by_frame[-2]

    def predict_bbox_update(self):
        predict_coordinate = self.get_predict_bbox()

        if len(predict_coordinate) > 0:
            self.bbox_center_by_frame.append(predict_coordinate)

    def get_predict_bbox(self):
        delta_coord_list = list()
        len_bbox = len(self.bbox_center_by_frame)
        # 기존은 30프레임을 봄
        for i in range(max(len_bbox-15, 0), len_bbox):
            delta_coord_list.append(np.subtract(self.bbox_center_by_frame[i], self.bbox_center_by_frame[i - 1]))

        if len(delta_coord_list) > 0:
            mean_coord = np.mean(delta_coord_list, axis=0)
            return np.add(self.bbox_center_by_frame[-1], mean_coord)
        else:
            return delta_coord_list

    def set_diff_feature(self, feature):
        dist_feat = feature - self.feature
        dist_feat = np.dot(dist_feat, dist_feat)

        if dist_feat > 100:
            self.feature_list.append(feature)


file_info_list = list()
with open(gt_ann_path, 'r') as f:
    file_info_list = json.load(f)

if len(file_info_list) > 0:
    img_info = dict()
    for file_info in file_info_list:
        filename = file_info['filename']
        gt_bboxes = file_info['ann']['bboxes']

        img_info[filename] = gt_bboxes

if one_dir_checked:
    img_dirs = [img_dir_path]
else:
    img_dirs = [os.path.join(img_dir_path, filename) for filename in os.listdir(img_dir_path)]

dataset = 'iitp'
score_thr=0.8

# 차량, 오토바이는 330을 준다.
similarity_thr=330.0

person_coord_dist_thr=50.0
coord_dist_thr=300.0
colors = random_colors(150)

if is_pkl_save:
    pkl_output = open(result_info_save_path, 'wb')
    result_info_dict = dict()

if is_pkl_load:
    pkl_file = open(result_info_save_path, 'rb')
    result_info_dict = pickle.load(pkl_file)
    pkl_file.close()

with open(result_csv_path, 'w') as wf:
    pass

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

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]

            inds = scores > score_thr

            scores = scores[inds]
            bboxes = bboxes[inds, :-1]
            labels = labels[inds]
            img_feats = img_feats[inds]

        # if i == 78:
        #     exit()

        # CLASS = {"사람": 0, "소화기":1, "소화전":2, "차량":3, "자전거":4, "오토바이":5}
        if i == 0:
            for now_label_index, label in enumerate(labels):
                objects_counter[label] += 1

                feature = img_feats[now_label_index]
                bbox = bboxes[now_label_index]
                score = scores[now_label_index]

                # label, feature, bbox, score, id
                object_info = ObjectInfo(label, feature, bbox, score, objects_counter[label])
                frame_dict[i][label].append(object_info)
        else:
            for now_label_index, label in enumerate(labels):
                # temp_dist = list()

                feature = img_feats[now_label_index]
                bbox = bboxes[now_label_index]
                score = scores[now_label_index]

                # 같은 라벨에 대해서 전부다 하는 것이 아닌 bbox가 가까운 아이들만 검사한다.
                # 좌표 가져오기
                x1, y1, x2, y2 = bbox

                more_before_dist = list()
                # 사람, 소화기, 소화전, 자전거 // 자동차,오토바이 따로 구분해서 처리
                if label in [0,1,2,4]:
                    # 원래 50 프레임
                    # max(i - 20, 0)
                    for before_frame_index in range(max(i - 60, 0), i):
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
                                dist_px = min(650.0, dist_frame * 200.0)
                            elif label == 1:
                                dist_px = min(650.0, dist_frame * 200.0)
                            elif label == 2:
                                dist_px = min(650.0, dist_frame * 200.0)
                            elif label == 4:
                                dist_px = min(650.0, dist_frame * 200.0)

                            center_x = (x2 + x1) / 2
                            center_y = (y2 + y1) / 2
                            # b_x1, b_y1, b_x2, b_y2 = before_object.bbox
                            # b_center_x = (b_x2 + b_x1) / 2
                            # b_center_y = (b_y2 + b_y1) / 2

                            # a = x1 - b_x1  # 선 a의 길이
                            # b = y1 - b_y1  # 선 b의 길이

                            # print((x1, y1), "//", (b_x1, b_y1), "//", before_object.id)
                            # 테스트용 코드 작성
                            pred_coord_dist = 1000.0
                            speed_x = 1000.0
                            speed_y = 1000.0
                            predict_coordinate = before_object.get_predict_bbox()
                            if len(predict_coordinate) > 0:
                                _a = center_x - predict_coordinate[0]  # 선 a의 길이
                                _b = center_y - predict_coordinate[1]  # 선 b의 길이

                                # _a = x1 - predict_coordinate[0]  # 선 a의 길이
                                # _b = y1 - predict_coordinate[1]  # 선 b의 길이
                                pred_coord_dist = math.sqrt((_a * _a) + (_b * _b))

                                speed_x, speed_y = np.abs(np.subtract(predict_coordinate, before_object.bbox_center_by_frame[-1]))

                            # 스피드가 빠르면 유사도를 조금 넉넉하게 준다.
                            alpha = 1.0
                            if speed_x > 30 or speed_y > 30:
                                alpha = 1.5

                            if label == 0 or label == 1 or label == 2:
                                # 90 이었음 // 140 사용함 지금은
                                limit_coord_dist = 100 * alpha
                            else:
                                limit_coord_dist = 150 * alpha

                            if dist_frame != 1:
                                for num in range(1, dist_frame + 1):
                                    limit_coord_dist += (limit_coord_dist * np.exp(-num)*3) #np.exp2(-num))

                            # edge_dist = 5
                            # if x1 < edge_dist or y1 < edge_dist or \
                            #     x2 > img_size[0] - edge_dist or y2 > img_size[1] - edge_dist and \
                            #         (img_size[1] - 100) > pred_coord_dist: # 왜냐하면 왼쪽테두리로 나갔는데 오른쪽 테두리에 나왔을 때를 가정
                            #     limit_coord_dist *= 1.2

                            # bbox가 최대치 일때 비교와 이전 프레임의 feature 비교를 한다.
                            # 넓이는 최대 100 px 까지는 여유로 둔다.
                            max_dist_feat = 1000.0
                            now_area = (x2 - x1) * (y2 - y1)
                            before_max_area = before_object.max_bbox_size[0] * before_object.max_bbox_size[1]
                            if max((before_max_area - 100), 0) <= now_area:
                                max_dist_feat = feature - before_object.max_bbox_feature
                                max_dist_feat = np.dot(max_dist_feat, max_dist_feat)

                            # 면적이 너무 차이나면 유사도 측정에 있어서 넉넉 하게 준다.
                            if abs(before_object.area - now_area) > 3000:
                                alpha += 0.5
                            # 또한 사람들 제외한 객체들은 유사도를 넉넉 하게 준다.
                            if label != 0:
                                alpha += 0.5

                            _dist_feat = feature - before_object.feature
                            _dist_feat = np.dot(_dist_feat, _dist_feat)

                            dist_feat_list = [_dist_feat]
                            for _index, before_freature in enumerate(before_object.feature_list):
                                _dist_feat = feature - before_freature
                                _dist_feat = np.dot(_dist_feat, _dist_feat)

                                # 임시로.. 오래된 이미지는 피쳐 신뢰도가 높을 수가 있으므로 약간 가중치를 준다.
                                _dist_feat = _dist_feat * np.exp(-_index)
                                dist_feat_list.append(_dist_feat)

                            dist_feat = min(dist_feat_list)

                            # 일단은 주위 100px 사이로만 검색 한다. (드론 이동이 있음) // 15 프레임까지는 거리 비교를 해본다.
                            # 또한 너무 비슷 하다면 기회를(?) 줘본다.
                            # print(dist_feat, max_dist_feat, dist_frame, "//", before_object.id, "//", pred_coord_dist, "//", b_x1, b_y1, "//", x1, y1)
                            # dist_px = dist_px if dist_frame < 15 else 100
                            # if min(x1+dist_px, img_size[0]) >= b_x1 and max(x1-dist_px,0) <= b_x1 and \
                            #         min(y1 + dist_px, img_size[1]) >= b_y1 and max(y1 - dist_px, 0) <= b_y1 or \
                            #         (pred_coord_dist < 130 and dist_frame < 30):

                            # if i == 259:
                            #     exit()

                            # 원래 13, 9 임
                            # print(dist_feat, max_dist_feat, dist_frame, "//", before_object.id, "//", pred_coord_dist,
                            #       "//", b_x1, b_y1, "//", x1, y1)
                            # if (dist_frame < 3 and min(center_x + dist_px, img_size[0]) >= b_center_x and max(center_x - dist_px, 0) <= b_center_x and \
                            #     min(center_y + dist_px, img_size[1]) >= b_center_y and max(center_y - dist_px, 0) <= b_center_y) or \
                            #         pred_coord_dist < limit_coord_dist:
                            if ((dist_frame < 3 and pred_coord_dist < dist_px) or pred_coord_dist < limit_coord_dist):
                                # boundary 안에 있더라도 너무 안닮았으면 패스 한다. 200 이었음
                                if dist_feat < 100*alpha: #25*alpha:
                                    more_before_dist.append([dist_feat*pred_coord_dist, before_frame_index, object_index])
                            elif dist_feat < 6*alpha:
                                more_before_dist.append([dist_feat*pred_coord_dist, before_frame_index, object_index])
                            elif max_dist_feat < 5:
                                more_before_dist.append([dist_feat*pred_coord_dist, before_frame_index, object_index])
                else:
                    # 원본 max(i - 5, 0) 자동차 30프레임이였음
                    for before_frame_index in range(max(i - 80, 0), i):
                        # print("index:", before_frame_index)
                        more_before_objects = frame_dict[before_frame_index]

                        more_before_label_objects = more_before_objects[label]
                        for object_index, before_object in enumerate(more_before_label_objects):
                            b_x1, b_y1 = before_object.bbox[:2]

                            # a = x1 - b_x1  # 선 a의 길이
                            # b = y1 - b_y1  # 선 b의 길이
                            #
                            # dist_feat = feature - before_object.feature
                            # dist_feat = np.dot(dist_feat, dist_feat)
                            #
                            # coord_dist = math.sqrt((a * a) + (b * b))

                            dist_frame = i - before_frame_index
                            # if label == 3:
                            #     dist_px = min(650.0, dist_frame * 250.0)
                            # else:
                            #     dist_px = min(650.0, dist_frame * 250.0)

                            # 테스트용 코드 작성
                            is_out_line = False
                            pred_coord_dist = 1500.0
                            predict_coordinate = before_object.get_predict_bbox()
                            if len(predict_coordinate) > 0:
                                pred_x = predict_coordinate[0]
                                pred_y = predict_coordinate[1]

                                center_x = (x2+x1)/2
                                center_y = (y2+y1)/2
                                _a = center_x - pred_x  # 선 a의 길이
                                _b = center_y - pred_y  # 선 b의 길이

                                # _a = x1 - predict_coordinate[0]  # 선 a의 길이
                                # _b = y1 - predict_coordinate[1]  # 선 b의 길이
                                pred_coord_dist = math.sqrt((_a * _a) + (_b * _b))

                                # 차량 같은 경우 테두리를 아예 벗어나면 드론이 직접 돌리지 않는 이상 되돌아오지 않는다.
                                max_bbox_size = before_object.max_bbox_size
                                half_w = max_bbox_size[0] / 2
                                half_h = max_bbox_size[1] / 2

                                # pred 오차를 생각하여 100 px 이 벗어나면 다시 보이는 차들은 새로운 차임
                                line_limit = 100
                                if pred_x < 0 and pred_x + half_w <= 0 - line_limit:
                                    is_out_line = True
                                elif pred_x > img_size[0] and pred_x - half_w >= img_size[0]+line_limit:
                                    is_out_line = True
                                elif pred_y < 0 and pred_y + half_h <= 0 - line_limit:
                                    is_out_line = True
                                elif pred_y > img_size[1] and pred_y - half_h >= img_size[1]+line_limit:
                                    is_out_line = True

                            # 240 원래
                            limit_coord_dist = 220

                            if dist_frame != 1:
                                for num in range(1, dist_frame + 1):
                                    limit_coord_dist += (limit_coord_dist * np.exp(-num)*3)#np.exp2(-num))

                            limit_coord_dist = min(500.0, limit_coord_dist)

                            edge_dist = 8
                            if x1 < edge_dist or y1 < edge_dist or \
                                x2 > img_size[0] - edge_dist or y2 > img_size[1] - edge_dist and \
                                    (img_size[1] - 100) > pred_coord_dist: # 왜냐하면 왼쪽테두리로 나갔는데 오른쪽 테두리에 나왔을 때를 가정:
                                limit_coord_dist *= 1.3

                            # bbox가 최대치 일때 비교와 이전 프레임의 feature 비교를 한다.
                            # 넓이는 최대 25*25 = 625 px 까지는 여유로 둔다.
                            max_dist_feat = 1000.0
                            before_max_area = before_object.max_bbox_size[0] * before_object.max_bbox_size[1]
                            if max((before_max_area - 625), 0) <= (x2 - x1) * (y2 - y1):
                                max_dist_feat = feature - before_object.max_bbox_feature
                                max_dist_feat = np.dot(max_dist_feat, max_dist_feat)

                            # dist_feat = feature - before_object.feature
                            # dist_feat = np.dot(dist_feat, dist_feat)
                            _dist_feat = feature - before_object.feature
                            _dist_feat = np.dot(_dist_feat, _dist_feat)

                            dist_feat_list = [_dist_feat]
                            for _index, before_freature in enumerate(before_object.feature_list):
                                _dist_feat = feature - before_freature
                                _dist_feat = np.dot(_dist_feat, _dist_feat)

                                # 임시로.. 오래된 이미지는 피쳐 신뢰도가 높을 수가 있으므로 약간 가중치를 준다.
                                _dist_feat = _dist_feat * np.exp(-_index)
                                dist_feat_list.append(_dist_feat)

                            dist_feat = min(dist_feat_list)

                            # 일단은 주위 100px 사이로만 검색 한다. (드론 이동이 있음) // 15 프레임까지는 거리 비교를 해본다.
                            # 또한 너무 비슷 하다면 기회를(?) 줘본다.
                            # print((x1, y1), "//", (b_x1, b_y1), "//", before_object.id, "//", dist_feat, max_dist_feat)
                            # print(dist_feat, max_dist_feat, dist_frame, "//", before_object.id, "//", pred_coord_dist)
                            # dist_px = dist_px if dist_frame < 10 else 300
                            # if min(x1 + dist_px, img_size[0]) >= b_x1 and max(x1 - dist_px, 0) <= b_x1 and \
                            #         min(y1 + dist_px, img_size[1]) >= b_y1 and max(y1 - dist_px, 0) <= b_y1  or \
                            #         (pred_coord_dist < limit_coord_dist and dist_frame < 30):

                            # 원래 feat 5, 3 이였음 지금 너무 피팅되서 숫자가 작게 나옴
                            if (dist_frame <= 4 and min(x1 + limit_coord_dist, img_size[0]) >= b_x1 and max(x1 - limit_coord_dist, 0) <= b_x1 and \
                                    min(y1 + limit_coord_dist, img_size[1]) >= b_y1 and max(y1 - limit_coord_dist, 0) <= b_y1)  or \
                                    pred_coord_dist < limit_coord_dist and not (is_out_line and dist_frame > 5):
                                # boundary 안에 있더라도 너무 안닮았으면 패스 한다.
                                # 원래 180
                                if dist_feat < 150:
                                    more_before_dist.append([dist_feat * pred_coord_dist, before_frame_index, object_index])
                            elif dist_feat < 1:
                                more_before_dist.append(
                                    [dist_feat * pred_coord_dist, before_frame_index, object_index])
                            elif max_dist_feat < 1:
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
        predictions['bboxes'] = bboxes
        predictions['scores'] = scores
        predictions['labels'] = labels

        print(objects_counter)
        if is_visualize:
            visualize(os.path.join(dir_path, filename), mmcv.imread(file_path), predictions, colors=colors, mask_display=False, class_names=class_names)

    if is_pkl_load:
        for i in range(len(imgs)):
            print(i, imgs[i])

            file_path = imgs[i]
            filename = os.path.split(file_path)[-1]

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