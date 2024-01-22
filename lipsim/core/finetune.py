import logging
import os
import pprint
import socket
import time
from os.path import join, exists

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from lipsim.core import utils
from lipsim.core.data.bapps_dataset import BAPPSDataset
from lipsim.core.data.night_dataset import NightDataset
from lipsim.core.data.readers import readers_config
from lipsim.core.evaluate import Evaluator
from lipsim.core.models.l2_lip.model import NormalizedModel, PerceptualMetric, L2LipschitzNetwork
from lipsim.core.models.l2_lip.model_v2 import L2LipschitzNetworkV2
from lipsim.core.trainer import Trainer
from lipsim.core.utils import N_CLASSES

# from core.models.dreamsim.model import dreamsim

os.environ["NCCL_DEBUG"] = "INFO"


class Finetuner(Trainer, Evaluator):

    def __init__(self, config):
        super(Finetuner, self).__init__(config)

    def __call__(self):

        cudnn.benchmark = True
        self.train_dir = self.config.train_dir
        self.ngpus = torch.cuda.device_count()

        self.rank = 0
        self.local_rank = 0
        self.num_nodes = 1
        self.num_tasks = 1
        self.is_master = True

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

        if self.config.dataset == 'night':
            data_loader, _ = NightDataset(config=self.config, batch_size=self.config.batch_size,
                                          split='train').get_dataloader()
        else:
            data_loader, _ = BAPPSDataset(data_dir=self.config.data_dir, load_size=224,
                                          split='train', dataset='traditional', make_path=True).get_dataloader(
                batch_size=self.config.batch_size)
        if self.local_rank == 0:
            logging.info(f"Using dataset: {self.config.dataset}")
        self.n_classes = N_CLASSES[self.config.teacher_model_name]
        # load model
        if self.config.target == 'lipsim_v2':
            self.model = L2LipschitzNetworkV2(self.config, self.n_classes)
        else:
            self.model = L2LipschitzNetwork(self.config, self.n_classes)
        self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)

        self.model = self.model.cuda()
        self.perceptual_metric = PerceptualMetric(backbone=self.model, requires_bias=self.config.requires_bias)
        param_size = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        if self.local_rank == 0:
            logging.info(f'Number of parameters to train: {param_size}')

        if self.is_distributed:
            utils.setup_distributed_training(self.world_size, self.rank)
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.local_rank == 0:
                logging.info('Model defined with DistributedDataParallel')
        else:
            self.model = nn.DataParallel(self.model, device_ids=[0])  # range(torch.cuda.device_count()))

        self.optimizer = utils.get_optimizer(self.config, self.model.parameters())
        self.saved_ckpts = set([0])
        self._load_state()
        # define set for saved ckpt

        sampler = None
        if sampler is not None:
            assert sampler.num_replicas == self.world_size

        if self.is_distributed:
            n_files = sampler.num_samples
        else:
            n_files = self.reader.n_train_files

        # define optimizer

        # define learning rate scheduler
        num_steps = self.config.epochs * (self.reader.n_train_files // self.global_batch_size)
        self.scheduler, self.warmup = utils.get_scheduler(self.optimizer, self.config, num_steps)
        # self.scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer=self.optimizer, max_lr=self.config.lr, min_lr=0.0, first_cycle_steps=num_steps, warmup_steps=200)

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

        epoch_id = 0
        self.optimizer.zero_grad()
        for epoch_id in range(start_epoch, self.config.epochs):
            if self.is_distributed:
                sampler.set_epoch(epoch_id)
            global_step, epoch = self.one_epoch_finetuning(data_loader, epoch_id, global_step)
            if self.config.dataset == 'night':
                self.certified_eval_for_night()
            else:
                self.lpips_eval()
        self._save_ckpt(global_step, epoch_id, final=True)
        logging.info("Done training -- epoch limit reached.")


    def one_epoch_finetuning(self, data_loader, epoch_id, global_step):

        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if epoch > epoch_id + 1:
                break
            img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                img_right.cuda(), target.cuda()

            start_time = time.time()
            epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
            dist_0, dist_1, _ = self.perceptual_metric.get_distance_between_images(img_ref, img_left, img_right,
                                                                                   requires_grad=True,
                                                                                   requires_normalization=True)
            logit = dist_0 - dist_1
            loss = self.criterion(logit.squeeze(), target)
            loss.backward()
            self.process_gradients(global_step)
            self.optimizer.step()
            self.scheduler.step(global_step)
            self.optimizer.zero_grad()
            seconds_per_batch = time.time() - start_time
            examples_per_second = self.global_batch_size / seconds_per_batch
            examples_per_second *= self.world_size

            self._save_ckpt(global_step, epoch_id)
            if global_step == 20 and self.is_master:
                self._print_approximated_train_time(start_time)
            global_step += 1
            self.log_training(epoch, epoch_id, examples_per_second, global_step, loss, start_time)


        return global_step, epoch

    def log_training(self, epoch, epoch_id, examples_per_second, global_step, loss, start_time):
        if self._to_print(global_step):
            lr = self.optimizer.param_groups[0]['lr']
            self.message.add("epoch", epoch, format="4.2f")
            self.message.add("step", global_step, width=5, format=".0f")
            self.message.add("lr", lr, format=".6f")
            self.message.add("loss", loss, format=".4f")
            if self.config.print_grad_norm:
                grad_norm = self.compute_gradient_norm()
                self.message.add("grad", grad_norm, format=".4f")
            self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
            logging.info(self.message.get_message())
        self._save_ckpt(global_step, epoch_id)
        if global_step == 20 and self.is_master:
            self._print_approximated_train_time(start_time)

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
            try:
                state = {
                    'epoch': epoch,
                    'global_step': step,
                    'model_state_dict': self.model.state_dict(),
                    'bias': self.perceptual_metric.bias,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }
            except Exception as e:
                print(e)
                state = {
                    'epoch': epoch,
                    'global_step': step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
            logging.debug("Saving checkpoint '{}'.".format(ckpt_name))
            torch.save(state, ckpt_path)
