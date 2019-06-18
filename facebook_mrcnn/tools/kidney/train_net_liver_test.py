# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time

import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tensorboardX import SummaryWriter


def last_checkpoint(save_path):
    last_checkpoint_log = os.path.join(save_path, 'last_checkpoint')

    if os.path.exists(last_checkpoint_log):
        with open(last_checkpoint_log, 'rt') as f:
            last_checkpoint = f.readline().strip('\n\r')
    else:
        last_checkpoint = ''

    return last_checkpoint


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    summary_writer = SummaryWriter(log_dir=output_dir)
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    if cfg.MODEL.WEIGHT.upper() == 'CONTINUE':
        model_weight = last_checkpoint(output_dir)
    else:
        model_weight = cfg.MODEL.WEIGHT
    extra_checkpoint_data = checkpointer.load(model_weight)

    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    data_loader_val = make_data_loader(
        cfg,
        is_train=False,
        is_distributed=distributed)[0]

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model=model,
        data_loader=data_loader,
        data_loader_val=data_loader_val,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer,
        device=device,
        checkpoint_period=checkpoint_period,
        arguments=arguments,
        summary_writer=summary_writer
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="../configs/kidney/e2e_mask_rcnn_X_101_32x8d_FPN_1x_liver_augmentation_test.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.OUTPUT_SUB_DIR:
        output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_SUB_DIR)
    else:
        now = time.localtime()
        time_dir_name = "%04d%02d%02d-%02d%02d%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        output_dir = os.path.join(cfg.OUTPUT_DIR, time_dir_name)
    cfg.merge_from_list(["OUTPUT_DIR", output_dir])

    cfg.freeze()

    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
