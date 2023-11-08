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


# def get_open_port():
#     import socket
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind(("", 0))
#     s.listen(1)
#     port = s.getsockname()[1]
#     s.close()
#     return port

def get_port_number():
    from socket import socket
    with socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    return port


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
    # host_name = stdout.decode().splitlines()[0]
    import platform
    host_name = platform.node()
    dist_url = f'tcp://{host_name}:{get_port_number()}'
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


class FeatureCrossEntropy(nn.Module):

    def __init__(self, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_out = student_output / self.student_temp
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        loss = 0
        for s_out in student_output:
            loss += torch.mean(-teacher_out * F.log_softmax(s_out / self.student_temp, dim=-1), dim=-1)
        print(loss)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def get_loss(config, margin=0, device='cuda:0'):
    if config.mode in ['lipsim', 'vanilla-eval']:
        return RMSELoss()
    elif config.mode == 'finetune':
        return HingeLoss(margin=config.margin, device=device)
    elif config.mode in ['train']:
        return FeatureCrossEntropy()


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


import sys
import time

import datetime
import torch
from collections import defaultdict, deque

import torch.distributed as dist


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        sys.stdout.flush()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
