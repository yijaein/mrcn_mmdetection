# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from facebook_mrcnn.maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler
from facebook_mrcnn.maskrcnn_benchmark.utils.model_serialization import load_state_dict
from facebook_mrcnn.maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from facebook_mrcnn.maskrcnn_benchmark.utils.imports import import_file
from facebook_mrcnn.maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.cfg = cfg

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            log_optimizer_scheduler_info(self.logger, self.optimizer, self.scheduler)

            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

        if self.cfg.PRIORITY_CONFIG:
            temp_optimizer = make_optimizer(self.cfg, self.model)
            self.optimizer.load_state_dict(temp_optimizer.state_dict())

            for group in self.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

            iteration = checkpoint['iteration'] if 'iteration' in checkpoint else 0
            last_epoch = iteration - 1
            temp_scheduler = make_lr_scheduler(self.cfg, self.optimizer, last_epoch=last_epoch)
            self.scheduler.load_state_dict(temp_scheduler.state_dict())

            # remove processed stat data
            for stat_name in ["optimizer", "scheduler"]:
                if stat_name in checkpoint:
                    checkpoint.pop(stat_name)
        else:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        if self.optimizer is not None and self.scheduler is not None:
            log_optimizer_scheduler_info(self.logger, self.optimizer, self.scheduler)

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            cfg, model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def log_optimizer_scheduler_info(logger, optimizer, scheduler):
        optimizer_stat_list = []
        optimizer_stat_dict = optimizer.state_dict()['param_groups'][0]
        optimizer_stat = [(key, value) for key, value in optimizer_stat_dict.items() if key not in ['params', 'lr']]
        for key, value in sorted(optimizer_stat, key=lambda l: l[0]):
            optimizer_stat_list.append('{:<15}\t{}'.format(key, value))

        scheduler_stat_list = []
        scheduler_stat_dict = scheduler.state_dict()
        scheduler_stat = [(key, value) for key, value in scheduler_stat_dict.items() if key not in ['base_lrs']]
        for key, value in sorted(scheduler_stat, key=lambda l: l[0]):
            scheduler_stat_list.append('{:<15}\t{}'.format(key, value))

        log = ''
        log += '\n' + 'Optimizer and Scheduler Stats'
        log += '\n' + 'Optimizer:'
        log += '\n\t' + '\n\t'.join(optimizer_stat_list)
        log += '\n' + 'Scheduler:'
        log += '\n\t' + '\n\t'.join(scheduler_stat_list)
        logger.info(log)
