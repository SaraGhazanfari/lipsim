import os
import re
import logging
import glob
import subprocess
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import lr_scheduler
import pytorch_warmup as warmup

import torch
from torchvision import transforms

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import random

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.
    else:
        if preprocess == "DEFAULT":
            t = transforms.Compose([
                transforms.Resize((load_size, load_size), interpolation=interpolation),
                transforms.ToTensor()
            ])
        elif preprocess == "DISTS":
            t = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        elif preprocess == "SSIM" or preprocess == "PSNR":
            t = transforms.ToTensor()
        else:
            raise ValueError("Unknown preprocessing method")
        return lambda pil_img: t(pil_img.convert("RGB"))


def get_epochs_from_ckpt(filename):
    regex = "(?<=ckpt-)[0-9]+"
    return int(re.findall(regex, filename)[-1])


def get_list_checkpoints(train_dir):
    files = glob.glob(join(train_dir, "checkpoints", "model.ckpt-*.pth"))
    files = sorted(files, key=get_epochs_from_ckpt)
    return [filename for filename in files]


class MessageBuilder:

    def __init__(self):
        self.msg = []

    def add(self, name, values, align=">", width=0, format=None):
        if name:
            metric_str = "{}: ".format(name)
        else:
            metric_str = ""
        values_str = []
        if type(values) != list:
            values = [values]
        for value in values:
            if format:
                values_str.append("{value:{align}{width}{format}}".format(
                    value=value, align=align, width=width, format=format))
            else:
                values_str.append("{value:{align}{width}}".format(
                    value=value, align=align, width=width))
        metric_str += '/'.join(values_str)
        self.msg.append(metric_str)

    def get_message(self):
        message = " | ".join(self.msg)
        self.clear()
        return message

    def clear(self):
        self.msg = []


def setup_logging(config, rank):
    level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
             'INFO': 20, 'WARN': 30
             }[config.logging_verbosity]
    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    filename = '{}/log_{}_{}.logs'.format(config.train_dir, config.mode, rank)
    f = open(filename, "a")
    logging.basicConfig(filename=filename, level=level, format=format_, datefmt='%H:%M:%S')


def setup_distributed_training(world_size, rank):
    """ find a common host name on all nodes and setup distributed training """
    # make sure http proxy are unset, in order for the nodes to communicate
    for var in ['http_proxy', 'https_proxy']:
        if var in os.environ:
            del os.environ[var]
        if var.upper() in os.environ:
            del os.environ[var.upper()]
    # get distributed url
    # cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    # stdout = subprocess.check_output(cmd.split())
    host_name = 'hostnames'  # stdout.decode().splitlines()[0]
    dist_url = f'tcp://{host_name}:9000'
    # setup dist.init_process_group
    dist.init_process_group(backend='nccl', init_method=dist_url,
                            world_size=world_size, rank=rank)


def get_loss(config):
    return nn.MSELoss()


def get_scheduler(optimizer, config, num_steps):
    """Return a learning rate scheduler schedulers."""
    if config.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)
    elif config.scheduler == 'interp':
        scheduler = TriangularLRScheduler(
            optimizer, num_steps, config.lr)
    elif config.scheduler == 'multi_step_lr':
        if config.decay is not None:
            steps_by_epochs = num_steps / config.epochs
            milestones = np.array(list(map(int, config.decay.split('-'))))
            milestones = list(np.int32(milestones * steps_by_epochs))
        else:
            milestones = list(map(int, [1 / 10 * num_steps, 5 / 10 * num_steps, 8.5 / 10 * num_steps]))
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=config.gamma)
    else:
        ValueError("Scheduler not reconized")
    warmup_scheduler = None
    if config.warmup_scheduler > 0:
        warmup_period = int(num_steps * config.warmup_scheduler)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    return scheduler, warmup_scheduler


def get_optimizer(config, params):
    """Returns the optimizer that should be used based on params."""
    lr, wd = config.lr, config.wd
    betas = (config.beta1, config.beta2)
    if config.optimizer == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=config.nesterov)
    elif config.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
    elif config.optimizer == 'adamw':
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    else:
        raise ValueError("Optimizer was not recognized")
    return opt


class TriangularLRScheduler:

    def __init__(self, optimizer, num_steps, lr):
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.lr = lr

    def step(self, t):
        lr = np.interp([t],
                       [0, self.num_steps * 2 // 5, self.num_steps * 4 // 5, self.num_steps],
                       [0, self.lr, self.lr / 20.0, 0])[0]
        self.optimizer.param_groups[0].update(lr=lr)
