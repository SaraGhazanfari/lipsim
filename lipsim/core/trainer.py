import os
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
from tqdm import tqdm
from dreamsim.model import download_weights, dreamsim
from lipsim.core import utils
from lipsim.core.attack.general_attack import GeneralAttack
from lipsim.core.data.bapps_dataset import BAPPSDataset
from lipsim.core.data.night_dataset import NightDataset
from lipsim.core.data.readers import readers_config, N_CLASSES

from lipsim.core.models.l2_lip.model import NormalizedModel, L2LipschitzNetwork, PerceptualMetric

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
        self.checkpoint = torch.load(path_last_ckpt)  # , map_location=self.model.device)
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
        self.ngpus = self.config.ngpus

        # job_env = submitit.JobEnvironment()
        self.rank = 0  # int(job_env.global_rank)
        self.local_rank = 0  # int(job_env.local_rank)
        self.num_nodes = 1  # int(job_env.num_nodes)
        self.num_tasks = 1  # int(job_env.num_tasks)
        self.is_master = bool(self.rank == 0)

        # Setup logging
        utils.setup_logging(self.config, self.rank)

        logging.info(self.rank)
        logging.info(self.local_rank)
        logging.info(self.num_nodes)
        logging.info(self.num_tasks)

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

        # distributed settings
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

        download_weights(cache_dir='./checkpoints', dreamsim_type=self.config.teacher_model_name)
        # self.teacher_model, _ = dreamsim(pretrained=True, dreamsim_type=self.config.teacher_model_name,
        #                                  cache_dir='./checkpoints')
        # self.teacher_model = self.teacher_model.cuda()
        # self.teacher_model.eval()

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
        self.perceptual_metric = PerceptualMetric(backbone=self.model)
        if self.config.dataset == 'bapps':
            data_loader, sampler = BAPPSDataset(data_dir=self.config.data_dir, load_size=224,
                                                split='train', dataset='traditional', make_path=True,
                                                is_distributed=self.is_distributed).get_dataloader(
                batch_size=self.config.batch_size)
        else:
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
            if self.config.mode == 'train':
                for n_batch, data in enumerate(data_loader):
                    if global_step == 2 and self.is_master:
                        start_time = time.time()
                    epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
                    self.one_step_training(data, epoch, global_step, epoch_id)

                    self._save_ckpt(global_step, epoch_id)
                    if global_step == 20 and self.is_master:
                        self._print_approximated_train_time(start_time)
                    global_step += 1
            else:
                global_step = self.one_epoch_finetuning(data_loader, epoch_id, global_step)

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

    def process_embedding(self, embeddings):
        if self.config.teacher_model_name == 'ensemble':
            return embeddings
        elif self.config.teacher_model_name == 'dino_vitb16':
            return embeddings[:, :768]
        elif self.config.teacher_model_name == 'open_clip_vitb32':
            return embeddings[:, 768:768 + 512]
        elif self.config.teacher_model_name == 'clip_vitb32':
            return embeddings[:, 768 + 512:]

    @torch.no_grad()
    def _teacher_model_embed(self, imgs):
        teacher_model, _ = dreamsim(pretrained=True, dreamsim_type=self.config.teacher_model_name,
                                    cache_dir='./checkpoints')
        teacher_model.eval()
        return teacher_model.embed(imgs)

    def one_step_training(self, data, epoch, step, epoch_id=0):
        self.optimizer.zero_grad()
        batch_start_time = time.time()
        images, embeddings = data
        images, _ = data
        embeddings = self.process_embedding(embeddings)
        original_imgs, jittered_imgs = images[:, 0, :, :], images[:, 1, :, :]
        original_imgs, jittered_imgs = original_imgs.cuda(), jittered_imgs.cuda()
        # embeddings = self._teacher_model_embed(original_imgs)
        embeddings = embeddings.cuda()
        if step == 0 and self.local_rank == 0:
            logging.info(f'images {original_imgs.shape}')
            logging.info(f'embeddings {embeddings.shape}')
        original_out = self.model(original_imgs)
        jittered_out = self.model(jittered_imgs)
        if step == 0 and self.local_rank == 0:
            logging.info(f'outputs {original_out.shape}')

        loss = self.criterion([original_out, jittered_out], embeddings,
                              epoch_id)  # + self.criterion(jittered_out, embeddings)
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

    def finetune_func(self):

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
        self.model = L2LipschitzNetwork(self.config, self.n_classes)
        self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)

        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # self.model = LipSimNetwork(self.config, n_classes=self.n_classes, backbone=self.backbone)

        self.model = self.model.cuda()
        self.perceptual_metric = PerceptualMetric(backbone=self.model)
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

        epoch_id = 0
        self.optimizer.zero_grad()
        for epoch_id in range(start_epoch, self.config.epochs):
            if self.is_distributed:
                sampler.set_epoch(epoch_id)
            global_step = self.one_epoch_finetuning(data_loader, epoch_id, global_step)
        self._save_ckpt(global_step, epoch_id, final=True)
        logging.info("Done training -- epoch limit reached.")
        if self.config.dataset == 'night':
            self.night_complete_eval()
        else:
            self.lpips_eval()

    def one_epoch_finetuning(self, data_loader, epoch_id, global_step):

        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                img_right.cuda(), target.cuda()

            start_time = time.time()
            epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
            dist_0, dist_1, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_left, img_right,
                                                                                       requires_grad=True,
                                                                                       requires_normalization=True)
            logit = dist_0 - dist_1
            loss = self.criterion(logit.squeeze(), target)
            loss.backward()
            self.process_gradients(global_step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            with self.warmup.dampening() if self.warmup else nullcontext():
                self.scheduler.step(global_step)
            seconds_per_batch = time.time() - start_time
            examples_per_second = self.global_batch_size / seconds_per_batch
            examples_per_second *= self.world_size

            self._save_ckpt(global_step, epoch_id)
            if global_step == 20 and self.is_master:
                self._print_approximated_train_time(start_time)
            global_step += 1
            self.log_training(epoch, epoch_id, examples_per_second, global_step, loss, start_time)
        self.certified_eval_for_lpips()
        return global_step

    @torch.no_grad()
    def certified_eval_for_lpips(self):

        for dataset in ['traditional', 'cnn']:  # , 'superres', 'deblur', 'color','frameinterp']:
            data_loader, _ = BAPPSDataset(data_dir=self.config.data_dir, load_size=224,
                                          split='val', dataset=dataset, make_path=True).get_dataloader(
                batch_size=self.config.batch_size)
            lpips_accuracy, lpips_certified = self.get_certified_accuracy(data_loader)

            eps_list = np.array([36, 72, 108])
            eps_float_list = eps_list / 255
            for i, eps_float in enumerate(eps_float_list):
                self.message.add('eps', eps_float, format='.5f')
                self.message.add(f'bapps accuracy {dataset}', lpips_accuracy[i], format='.5f')
                self.message.add(f'bapps certified {dataset}', lpips_certified[i], format='.5f')
                logging.info(self.message.get_message())

    @torch.no_grad()
    def lpips_eval(self):
        for dataset in ['traditional', 'cnn', 'superres', 'deblur', 'color',
                        'frameinterp']:
            data_loader, _ = BAPPSDataset(data_dir=self.config.data_dir, load_size=224,
                                          split='val', dataset=dataset, make_path=True).get_dataloader(
                batch_size=self.config.batch_size)
            twoafc_score = self.get_2afc_score_eval(data_loader)
            logging.info(f"BAPPS 2AFC score: {str(twoafc_score)}")
        return twoafc_score

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

    def night_complete_eval(self):
        data_loader, dataset_size = NightDataset(config=self.config, batch_size=self.config.batch_size,
                                                 split='test_imagenet').get_dataloader()
        no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(config=self.config,
                                                                         batch_size=self.config.batch_size,
                                                                         split='test_no_imagenet').get_dataloader()

        imagenet_accuracy, imagenet_certified = self.get_certified_accuracy(data_loader)
        torch.cuda.empty_cache()
        no_imagenet_accuracy, no_imagenet_certified = self.get_certified_accuracy(no_imagenet_data_loader)
        torch.cuda.empty_cache()
        overall_accuracy = (imagenet_accuracy * dataset_size + no_imagenet_accuracy * no_imagenet_dataset_size) / (
                dataset_size + no_imagenet_dataset_size)
        overall_certified = (imagenet_certified * dataset_size + no_imagenet_certified * no_imagenet_dataset_size) / (
                dataset_size + no_imagenet_dataset_size)
        eps_list = np.array([36, 72, 108, 255])
        eps_float_list = eps_list / 255
        for i, eps_float in enumerate(eps_float_list):
            self.message.add('eps', eps_float, format='.5f')
            self.message.add('imagenet accuracy', imagenet_accuracy[i], format='.5f')
            self.message.add('imagenet certified', imagenet_certified[i], format='.5f')

            self.message.add('no imagenet accuracy', no_imagenet_accuracy[i], format='.5f')
            self.message.add('no imagenet certified', no_imagenet_certified[i], format='.5f')

            self.message.add('Overall accuracy', overall_accuracy[i], format='.5f')
            self.message.add('Overall certified', overall_certified[i], format='.5f')
            logging.info(self.message.get_message())

    def get_certified_accuracy(self, data_loader):
        # self.model.eval()
        running_accuracy = np.zeros(4)
        running_certified = np.zeros(4)
        running_inputs = 0
        lip_cst = 2
        eps_list = np.array([36, 72, 108, 255])
        eps_float_list = eps_list / 255
        with torch.no_grad():
            for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
                img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                    img_right.cuda(), target.cuda()

                dist_0, dist_1, bound = self.perceptual_metric.get_cosine_score_between_images(img_ref,
                                                                                               img_left=img_left,
                                                                                               img_right=img_right,
                                                                                               requires_normalization=True)

                outputs = torch.stack((dist_1, dist_0), dim=1)
                predicted = outputs.argmax(axis=1)
                correct = outputs.max(1)[1] == target
                fy_fi = (outputs.max(dim=1)[0].reshape(-1, 1) - outputs)
                mask = (outputs.max(dim=1)[0].reshape(-1, 1) - outputs) == 0
                fy_fi[mask] = torch.inf
                radius = (fy_fi / bound).min(dim=1)[0]
                for i, eps_float in enumerate(eps_float_list):
                    certified = radius > eps_float
                    running_certified[i] += torch.sum(correct & certified).item()
                    running_accuracy[i] += predicted.eq(target.data).cpu().sum().numpy()
                running_inputs += img_ref.size(0)

        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs

        return accuracy, certified

    def get_2afc_score_eval(self, test_loader):
        logging.info("Evaluating NIGHTS dataset.")
        d0s = []
        d1s = []
        targets = []
        # with torch.no_grad()
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
                img_right.cuda(), target.cuda()
            dist_0, dist_1, target = self.one_step_2afc_score_eval(img_ref, img_left, img_right, target)
            d0s.append(dist_0)
            d1s.append(dist_1)
            targets.append(target)

        twoafc_score = get_2afc_score(d0s, d1s, targets)
        return twoafc_score

    def one_step_2afc_score_eval(self, img_ref, img_left, img_right, target):
        if self.config.attack:
            img_ref = GeneralAttack(self.config).generate_attack(img_ref, img_left, img_right, target,
                                                                 target_model=self.model_wrapper(img_left, img_right))
        dist_0, dist_1, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_left, img_right)
        if len(dist_0.shape) < 1:
            dist_0 = dist_0.unsqueeze(0)
            dist_1 = dist_1.unsqueeze(0)
        dist_0 = dist_0.unsqueeze(1)
        dist_1 = dist_1.unsqueeze(1)
        target = target.unsqueeze(1)
        return dist_0, dist_1, target

    def model_wrapper(self, img_left, img_right):
        def metric_model(img):
            if len(img.shape) > 4:
                img_ref, img_0, img_1 = img[:, 0, :, :].squeeze(1), img[:, 1, :, :].squeeze(1), img[:, 2, :, :].squeeze(
                    1)
            else:
                img_ref = img
                img_0, img_1 = img_left, img_right
            dist_0, dist_1, _ = self.perceptual_metric.get_cosine_score_between_images(img_ref, img_0, img_1,
                                                                                       requires_grad=True)
            return torch.stack((dist_1, dist_0), dim=1)

        return metric_model


def get_2afc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores)
    return twoafc_score
