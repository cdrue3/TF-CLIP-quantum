import os
import os.path as osp
import sys
import datetime

import scipy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import argparse
from config import cfg

from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2

class _LimitedLoader:
    def __init__(self, loader, max_batches):
        self._loader = loader
        self._max = max_batches
    def __len__(self):
        return min(self._max, len(self._loader))
    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._max:
                break
            yield batch
    def __getattr__(self, name):
        return getattr(self._loader, name)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    #############################################
    #--> 加载参数和初始化
    #############################################
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/vit_clipreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--max_batches", default=None, type=int,
                        help="Limit stage-2 loader to this many batches per epoch (smoke test).")
    parser.add_argument("--max_mem_batches", default=None, type=int,
                        help="Limit stage-1 CLIP-memory loader to this many batches.")
    parser.add_argument("--fast_schedule", action="store_true", default=False,
                        help="Scale LR decay steps proportionally to MAX_EPOCHS.")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable AMP (may be faster on GPUs without Tensor Cores).")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Apply torch.compile to the model for potential speedup.")
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    #############################################
    #--> 数据加载
    #############################################
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if args.max_mem_batches is not None:
        train_loader_stage1 = _LimitedLoader(train_loader_stage1, args.max_mem_batches)
    if args.max_batches is not None:
        train_loader_stage2 = _LimitedLoader(train_loader_stage2, args.max_batches)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    if args.compile:
        logger.info("Applying torch.compile to model...")
        model = torch.compile(model)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)

    sched_steps = list(cfg.SOLVER.STAGE2.STEPS)
    if args.fast_schedule:
        total = cfg.SOLVER.STAGE2.MAX_EPOCHS
        sched_steps = [max(1, int(total * 0.75)), max(2, int(total * 0.90))]
        logger.info(f"[fast_schedule] MAX_EPOCHS={total}, scaled steps={sched_steps}")

    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, sched_steps, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage1,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank,
        num_classes,
        use_amp=not args.no_amp,
    )