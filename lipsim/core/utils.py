import os
import re
import random
import logging
import glob
import subprocess
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import lr_scheduler
from torchvision import transforms
import pytorch_warmup as warmup
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
                radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.
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
    # format_ = "[%(asctime)s %(pathname)s:%(lineno)s] %(message)s"
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
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]
    dist_url = f'tcp://{host_name}:9000'
    # setup dist.init_process_group
    dist.init_process_group(backend='nccl', init_method=dist_url,
                            world_size=world_size, rank=rank)


class RMSELoss(nn.Module):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class HingeLoss(torch.nn.Module):

    def __init__(self, device, margin):
        super(HingeLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, x, y):
        y_rounded = torch.round(y)  # Map [0, 1] -> {0, 1}
        y_transformed = -1 * (1 - 2 * y_rounded)  # Map {0, 1} -> {-1, 1}
        return torch.max(torch.zeros(x.shape).to(self.device), self.margin + (-1 * (x * y_transformed))).sum()


def get_loss(config, margin=0, device='cuda:0'):
    if config.mode in ['train', 'lipsim', 'vanilla-eval']:
        return RMSELoss()
    elif config.mode == 'finetune':
        return HingeLoss(margin=config.margin, device=device)


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
