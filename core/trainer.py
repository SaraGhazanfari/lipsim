import os
import sys
import time
import pprint
import socket
import logging
import glob
from os.path import join, exists
from contextlib import nullcontext

import submitit
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.multiprocessing.errors import record

from core import utils
from core.models import N_CLASSES
from core.data.readers import readers_config
from core.models.l2_lip.model import NormalizedModel, L2LipschitzNetwork

# from core.models.dreamsim.model import dreamsim

os.environ["NCCL_DEBUG"] = "INFO"


class Trainer:
  """A Trainer to train a PyTorch."""

  def __init__(self, config):
      self.config = config

  def _load_state(self):
      # load last checkpoint
      checkpoints = glob.glob(join(self.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
      get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
      checkpoints = sorted(
          [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
      path_last_ckpt = join(self.train_dir, 'checkpoints', checkpoints[-1])
      self.checkpoint = torch.load(path_last_ckpt, map_location=self.model.device)
      self.model.load_state_dict(self.checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
      self.saved_ckpts.add(self.checkpoint['epoch'])
      epoch = self.checkpoint['epoch']
      if self.local_rank == 0:
          logging.info('Loading checkpoint {}'.format(checkpoints[-1]))

  def _save_ckpt(self, step, epoch, final=False, best=False):
      """Save ckpt in train directory."""
      freq_ckpt_epochs = self.config.save_checkpoint_epochs
      if (epoch % freq_ckpt_epochs == 0 and self.is_master and epoch not in self.saved_ckpts) or (
              final and self.is_master) or best:
          prefix = "model" if not best else "best_model"
          ckpt_name = f"{prefix}.ckpt-{step}.pth"
          ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
          if exists(ckpt_path) and not best:
              return
          self.saved_ckpts.add(epoch)
          state = {
              'epoch': epoch,
              'global_step': step,
              'model_state_dict': self.model.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
              # 'scheduler': self.scheduler.state_dict()
          }
          logging.debug("Saving checkpoint '{}'.".format(ckpt_name))
          torch.save(state, ckpt_path)

  @record
  def __call__(self):
    """Performs training and evaluation
    """
    cudnn.benchmark = True
    self.train_dir = self.config.train_dir
    self.ngpus = torch.cuda.device_count()

    job_env = submitit.JobEnvironment()
    self.rank = int(job_env.global_rank)
    self.local_rank = int(job_env.local_rank)
    self.num_nodes = int(job_env.num_nodes)
    self.num_tasks = int(job_env.num_tasks)
    self.is_master = bool(self.rank == 0)

    # Setup logging
    utils.setup_logging(self.config, self.rank)

    torch.cuda.init()

    self.message = utils.MessageBuilder()
    # print self.config parameters
    if self.local_rank == 0:
        logging.info(self.config.cmd)
        pp = pprint.PrettyPrinter(indent=2, compact=True)
        logging.info(pp.pformat(vars(self.config)))
    # print infos
    if self.local_rank == 0:
        logging.info(f"PyTorch version: {torch.__version__}.")
        logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
        logging.info(f"Hostname: {socket.gethostname()}.")

    # ditributed settings
    self.world_size = 1
    self.is_distributed = False
    if self.num_nodes > 1 or self.num_tasks > 1:
      self.is_distributed = True
      self.world_size = self.num_nodes * self.ngpus
    if self.num_nodes > 1:
      logging.info(f"Distributed Training on {self.num_nodes} nodes")
    elif self.num_nodes == 1 and self.num_tasks > 1:
      logging.info(f"Single node Distributed Training with {self.num_tasks} tasks")
    else:
      assert self.num_nodes == 1 and self.num_tasks == 1
      logging.info("Single node training.")

    if not self.is_distributed:
      self.batch_size = self.config.batch_size * self.ngpus
    else:
      self.batch_size = self.config.batch_size

    self.global_batch_size = self.batch_size * self.world_size
    logging.info('World Size={} => Total batch size {}'.format(
        self.world_size, self.global_batch_size))

    torch.cuda.set_device(self.local_rank)

    # load dataset
    Reader = readers_config[self.config.dataset]
    self.reader = Reader(config=self.config, batch_size=self.batch_size, is_training=True,
                         is_distributed=self.is_distributed)
    if self.local_rank == 0:
        logging.info(f"Using dataset: {self.config.dataset}")
    self.n_classes = N_CLASSES[self.config.teacher_model_name]

    # load model
    self.model = L2LipschitzNetwork(self.config, self.n_classes)
    self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)
    self.model = self.model.cuda()

    param_size = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
    if self.local_rank == 0:
      logging.info(f'Number of parameters to train: {param_size}')

    # self.teacher_model, _ = dreamsim(pretrained=True,
    #   dreamsim_type=self.config.teacher_model_name, cache_dir=self.config.dreamsim_path)
    # self.teacher_model = self.teacher_model.cuda()

    # setup distributed process if training is distributed
    # and use DistributedDataParallel for distributed training
    if self.is_distributed:
      utils.setup_distributed_training(self.world_size, self.rank)
      self.model = DistributedDataParallel(
        self.model, device_ids=[self.local_rank], output_device=self.local_rank)
      if self.local_rank == 0:
        logging.info('Model defined with DistributedDataParallel')
    else:
      self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

    # define set for saved ckpt
    self.saved_ckpts = set([0])

    data_loader, sampler = self.reader.load_dataset()
    if sampler is not None:
      assert sampler.num_replicas == self.world_size

    if self.is_distributed:
      n_files = sampler.num_samples
    else:
      n_files = self.reader.n_train_files

    # define optimizer
    self.optimizer = utils.get_optimizer(self.config, self.model.parameters())

    # define learning rate scheduler
    num_steps = self.config.epochs * (self.reader.n_train_files // self.global_batch_size)
    self.scheduler, self.warmup = utils.get_scheduler(self.optimizer, self.config, num_steps)
    if self.config.warmup_scheduler is not None:
      logging.info(f"Warmup scheduler on {self.config.warmup_scheduler * 100:.0f}% of training")

    # define the loss
    self.criterion = utils.get_loss(self.config)

    if self.local_rank == 0:
      logging.info("Number of files on worker: {}".format(n_files))
      logging.info("Start training")

    # training loop
    start_epoch, global_step = 0, 0
    self.best_checkpoint = None
    self.best_accuracy = None
    self.best_accuracy = [0., 0.]
    for epoch_id in range(start_epoch, self.config.epochs):
      if self.is_distributed:
        sampler.set_epoch(epoch_id)
      for n_batch, data in enumerate(data_loader):
        if global_step == 2 and self.is_master:
          start_time = time.time()
        epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
        self.one_step_training(data, epoch, global_step)
        self._save_ckpt(global_step, epoch_id)
        if global_step == 20 and self.is_master:
          self._print_approximated_train_time(start_time)
        global_step += 1

    self._save_ckpt(global_step, epoch_id, final=True)
    logging.info("Done training -- epoch limit reached.")

  def compute_gradient_norm(self):
    grad_norm = 0.
    for name, p in self.model.named_parameters():
      if p.grad is None: continue
      norm = p.grad.detach().data.norm(2)
      grad_norm += norm.item() ** 2
    grad_norm = grad_norm ** 0.5
    return grad_norm

  def _print_approximated_train_time(self, start_time):
    total_steps = self.reader.n_train_files * self.config.epochs / self.global_batch_size
    total_seconds = total_steps * ((time.time() - start_time) / 18)
    n_days = total_seconds // 86400
    n_hours = (total_seconds % 86400) / 3600
    logging.info(
        'Approximated training time: {:.0f} days and {:.1f} hours'.format(
            n_days, n_hours))

  def _to_print(self, step):
    frequency = self.config.frequency_log_steps
    if frequency is None:
      return False
    return (step % frequency == 0 and self.local_rank == 0) or \
      (step == 1 and self.local_rank == 0)

  def process_gradients(self, step):
    if self.config.gradient_clip_by_norm:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by norm: {}".format(
          self.config.gradient_clip_by_norm))
      torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config.gradient_clip_by_norm)
    elif self.config.gradient_clip_by_value:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by value: {}".format(
          self.config.gradient_clip_by_value))
      torch.nn.utils.clip_grad_value_(
        self.model.parameters(), self.config.gradient_clip_by_value)

  def one_step_training(self, data, epoch, step):
    self.optimizer.zero_grad()
    batch_start_time = time.time()
    images, embeddings = data
    original_imgs, jittered_imgs = images[:, 0, :, :], images[:, 1, :, :]
    original_imgs, jittered_imgs = original_imgs.cuda(), jittered_imgs.cuda()
    embeddings = embeddings.cuda()
    if step == 0 and self.local_rank == 0:
      logging.info(f'images {original_imgs.shape}')
      logging.info(f'embeddings {embeddings.shape}')
    original_out = self.model(original_imgs)
    jittered_out = self.model(jittered_imgs)
    if step == 0 and self.local_rank == 0:
      logging.info(f'outputs {original_out.shape}')
    # teacher_out = self.teacher_model.embed(original_imgs)
    loss = self.criterion(original_out, embeddings) + self.criterion(jittered_out, embeddings)
    loss.backward()
    self.process_gradients(step)
    self.optimizer.step()
    with self.warmup.dampening() if self.warmup else nullcontext():
      self.scheduler.step(step)
    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.global_batch_size / seconds_per_batch
    examples_per_second *= self.world_size
    if self._to_print(step):
      lr = self.optimizer.param_groups[0]['lr']
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", lr, format=".6f")
      self.message.add("loss", loss, format=".4f")
      if self.config.print_grad_norm:
        grad_norm = self.compute_gradient_norm()
        self.message.add("grad", grad_norm, format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
      logging.info(self.message.get_message())



