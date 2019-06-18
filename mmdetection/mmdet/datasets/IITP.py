import mmcv
import numpy as np
import os.path as osp

from mmcv.parallel import DataContainer as DC
from .utils import to_tensor, random_scale

from .custom import CustomDataset


class IITPDataset(CustomDataset):
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
    # CLASS = {"사람": 1, "소화기":2, "소화전":3, "차량":4, "자전거":5, "오토바이":6}
    CLASSES = ("person", "extinguisher", "hydrant", "car", "bicycle", "bike")

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

        return ann_info

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # img = mmcv.imresize(img, (1024,576))

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']
        if self.with_mask:
            gt_masks = ann['masks']
        else:
            gt_masks = list()

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # 현재는 bbox/mask 를 augment를 통해 사라진다면 원본을 사용 한다.
        if self.custom_aug is not None:
            flip = False
            img, gt_bboxes, gt_masks = self._do_custom_augmentation(img, gt_bboxes, gt_masks)
            h, w = img.shape[:2]

            if w < 1024:
                img_scale = (h, w)
            else:
                img_scale = random_scale(self.img_scales)  # sample a scale
        else:
            # extra augmentation
            if self.extra_aug is not None:
                img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            img_scale = random_scale(self.img_scales)  # sample a scale

        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(gt_masks, pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        img = mmcv.imresize(img, (1024, 576))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data