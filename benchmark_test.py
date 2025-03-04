# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import random
import queue
import threading
from collections.abc import MutableMapping, Sequence

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from train_net_video import Trainer, setup


def main(args):
    torch.backends.cudnn.deterministic = True

    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    model = model.eval()

    data_loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])

    for idx, inputs in enumerate(data_loader):
        with torch.amp.autocast("cuda"), torch.no_grad():
            # directly inference
            outputs = model(inputs)
        
        if idx == 5:
            break

    import time
    avg_time = []
    for idx, inputs in enumerate(data_loader):
        with torch.amp.autocast("cuda"), torch.no_grad():
            # online inference
            images = sum([x["image"] for x in inputs], [])

            video_outputs = [None]
            for idx, image in enumerate(images):
                image = image.cuda()
                start = time.time()
                video_output = model.online_inference(image[None], video_outputs[-1])
                end = time.time()
                video_outputs.append(video_output)
                avg_time.append(end - start)
                if len(avg_time) > 50:
                    print("frame time:", sum(avg_time[50:]) / len(avg_time[50:]))
            video_outputs = video_outputs[1:]

        if idx == 10:
            break

# def main(args):
#     cfg = setup(args)

#     model = Trainer.build_model(cfg)
#     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

#     model = model.eval()

#     pseudo_input = [
#         {
#             "image": [torch.rand(3, 720, 1280).cuda()] * 40,
#             "height": 720,
#             "width":  1280,
#         },
#         # {
#         #     "image": [torch.rand(3, 727, 1236).cuda()] * 40,
#         #     "height": 727,
#         #     "width":  1236,
#         # }
#     ]

#     # AMP context
#     for _ in range(10):
#         with torch.amp.autocast("cuda"), torch.no_grad():
#             outputs = model(pseudo_input)

#     # with torch.amp.autocast("cuda"), torch.no_grad():
#     #     for i in range(4):
#     #         outputs = model.online_inference(torch.randn(1, 5, 3, 720, 1280).cuda())

#     exit(0)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
