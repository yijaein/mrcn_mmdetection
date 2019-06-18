from __future__ import division

import argparse
from mmcv import Config

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from mmdetection.mmdet import __version__
from mmdetection.mmdet.datasets import get_dataset
from mmdetection.mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdetection.mmdet.models import build_detector
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # 불필요한 loss 와 학습에서 제외 할 때 사용
    if 'no_train_modules' in cfg.train_cfg:
        no_train_modules = cfg.train_cfg.no_train_modules
        if len(no_train_modules) != 0:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.neck.parameters():
                param.requires_grad = False

        if 'rpn_haed' in no_train_modules:
            for param in model.rpn_head.parameters():
                param.requires_grad = False
        elif 'bbox_head' in no_train_modules:
            for param in model.bbox_head.parameters():
                param.requires_grad = False
        elif 'mask_head' in no_train_modules and hasattr(cfg.model, 'mask_head'):
            for param in model.mask_head.parameters():
                param.requires_grad = False

    train_dataset = get_dataset(cfg.data.train)

    from imgaug import augmenters as iaa
    # augmentation = iaa.Sequential([
    #     iaa.Fliplr(0.5),
    #     # iaa.Invert(0.5),
    #     iaa.Affine(
    #         scale={"x": (0.7, 1.0), "y": (0.7, 1.0)},
    #         translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
    #     ),
    #     iaa.Multiply((0.5, 1.0))
    # ], random_order=True)

    augmentation = iaa.SomeOf((0, None), [
        iaa.Fliplr(0.5),
        # iaa.Invert(0.5),
        iaa.Affine(
            scale={"x": (0.7, 1.0), "y": (0.7, 1.0)},
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        ),
        iaa.Multiply((0.4, 1.0))
    ])

    train_dataset.set_augmentation(augmentation)

    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
