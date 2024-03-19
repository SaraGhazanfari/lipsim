import glob
import logging
import os
import pprint
import socket
from os.path import join, exists

import submitit
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from lipsim.core import utils
from lipsim.core.cosine_scheduler import CosineAnnealingWarmupRestarts
from lipsim.core.models.l2_lip.model import NormalizedModel, ClassificationLayer, LipschitzClassifier
from lipsim.core.models.l2_lip.model_v2 import L2LipschitzNetworkV2
from lipsim.core.utils import N_CLASSES


class LinearEvaluation:
    def __init__(self, config):
        self.config = config
        self.train_dir = self.config.train_dir

    def _init_class_properties(self):
        job_env = submitit.JobEnvironment()
        self.rank = int(job_env.global_rank)
        self.local_rank = int(job_env.local_rank)
        self.num_nodes = int(job_env.num_nodes)
        self.num_tasks = int(job_env.num_tasks)
        self.is_master = bool(self.local_rank == 0)
        self.ngpus = self.config.ngpus
        self.world_size = self.num_nodes * self.ngpus
        self.embed_dim = N_CLASSES[self.config.teacher_model_name]

        self.message = utils.MessageBuilder()
        utils.setup_logging(self.config, 0)
        logging.info(self.rank)
        logging.info(self.local_rank)
        logging.info(self.num_nodes)
        logging.info(self.num_tasks)

        torch.cuda.init()

        utils.setup_distributed_training(self.world_size, self.rank, self.config.dist_url)
        if self.local_rank == 0:
            logging.info(self.config.cmd)
            pp = pprint.PrettyPrinter(indent=2, compact=True)
            logging.info(pp.pformat(vars(self.config)))
            logging.info(f"PyTorch version: {torch.__version__}.")
            logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
            logging.info(f"Hostname: {socket.gethostname()}.")

        means = (0.0000, 0.0000, 0.0000)
        stds = (1.0000, 1.0000, 1.0000)
        model = L2LipschitzNetworkV2(self.config, self.embed_dim)

        self.model = NormalizedModel(model, means, stds)

        if self.local_rank == 0:
            self.model = self.load_ckpt()
        torch.cuda.set_device(self.local_rank)
        self.linear_classifier = ClassificationLayer(self.config,
                                                     embed_dim=N_CLASSES[self.config.teacher_model_name],
                                                     n_classes=1000)
        if self.local_rank == 0:
            param_size = utils.get_parameter_number(self.linear_classifier)
            logging.info(f'Number of parameters to train: {param_size}')

        logging.info(f"Distributed Training on {self.local_rank} gpus")
        self.lipschitz_classifier = LipschitzClassifier(backbone=self.model, classifier=self.linear_classifier).cuda()
        self.lipschitz_classifier = DistributedDataParallel(self.lipschitz_classifier, device_ids=[self.local_rank],
                                                            output_device=self.local_rank)

        self.optimizer = utils.get_optimizer(self.config, self.lipschitz_classifier.parameters())
        if self.local_rank == 0:
            logging.info(f'Optimizer created for the classifier')
        standard_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.train_dataset = ImageFolder(root=os.path.join(self.config.data_dir, 'train'), transform=standard_transform)
        val_dataset = ImageFolder(root=os.path.join(self.config.data_dir, 'val'), transform=standard_transform)
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=4,
                                       shuffle=False, pin_memory=False, sampler=self.sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=4, shuffle=False,
                                     pin_memory=False)

        if self.local_rank == 0:
            logging.info(f"Using dataset: {self.config.dataset} with size:{len(self.train_dataset)}")

    def _save_ckpt(self, epoch, final=False, best=False):
        """Save ckpt in train directory."""
        freq_ckpt_epochs = self.config.save_checkpoint_epochs
        if (epoch % freq_ckpt_epochs == 0 and self.is_master and epoch not in self.saved_ckpts) or (
                final and self.is_master) or best:
            prefix = "model" if not best else "best_model"
            ckpt_name = f"{prefix}.ckpt-{epoch}.pth"
            ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
            if exists(ckpt_path) and not best:
                return
            self.saved_ckpts.add(epoch)
            state = {
                'epoch': epoch,
                # 'global_step': step,
                'model_state_dict': self.lipschitz_classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict()
            }
            torch.save(state, ckpt_path)

    def __call__(self):
        self._init_class_properties()
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(self.config)).items())))
        cudnn.benchmark = True
        self.saved_ckpts = set([0])
        num_steps = (self.config.epochs * len(self.train_dataset) // (self.config.batch_size * self.world_size))
        self.scheduler = CosineAnnealingWarmupRestarts(optimizer=self.optimizer, max_lr=self.config.lr,
                                                       min_lr=0,
                                                       first_cycle_steps=num_steps,
                                                       warmup_steps=num_steps * 5 / self.config.epochs)
        for epoch in range(0, self.config.epochs):
            self.sampler.set_epoch(epoch)
            self.train(epoch)

            if (epoch % self.config.frequency_log_steps == self.config.frequency_log_steps - 1
                or epoch == self.config.epochs - 1) and self.local_rank == 0:
                acc1, loss = self.evaluate()
                self.message.add("epoch", epoch, format="4.2f")
                self.message.add("loss", loss, format=".4f")
                self.message.add("acc", acc1, format=".4f")
                logging.info(self.message.get_message())
            self._save_ckpt(epoch=epoch)
        self._save_ckpt(epoch=self.config.epochs, final=True)

    def train(self, epoch):
        num_steps = len(self.train_dataset) // (self.config.batch_size * self.world_size)
        self.linear_classifier.train()
        for idx, (inp, target) in tqdm(enumerate(self.train_loader)):
            self.optimizer.zero_grad()
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = self.lipschitz_classifier(inp)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.process_gradients()
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.synchronize()

            if idx % int(num_steps / 100) == int(num_steps / 100) - 1 and self.local_rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.message.add("epoch_id", epoch)
                self.message.add("epoch", idx / len(self.train_loader), format="4.2f")
                self.message.add("step", idx + 1, width=5, format=".0f")
                self.message.add("lr", lr, format=".6f")
                self.message.add("loss", loss, format=".4f")
                grad_norm = self.compute_gradient_norm()
                self.message.add("grad", grad_norm, format=".4f")
                logging.info(self.message.get_message())

    @torch.no_grad()
    def evaluate(self):
        self.linear_classifier.eval()
        for inp, target in self.val_loader:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = self.lipschitz_classifier(inp)
            loss = nn.CrossEntropyLoss()(output, target)
            acc1, = utils.accuracy(output, target, topk=(1,))
        return acc1, loss

    def compute_gradient_norm(self):
        grad_norm = 0.
        for name, p in self.model.named_parameters():
            if p.grad is None: continue
            norm = p.grad.detach().data.norm(2)
            grad_norm += norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        return grad_norm

    def load_ckpt(self):
        checkpoints = glob.glob(join(self.config.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
        ckpt_path = join(self.config.train_dir, 'checkpoints', ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'alpha' not in k:
                new_checkpoint[k] = v
            if 'module' in k:
                new_checkpoint[k.replace('module.', '')] = v
                del new_checkpoint[k]
        msg = self.model.load_state_dict(new_checkpoint)
        logging.info(f'The checkpoint {ckpt_name} was loaded with {msg}')
        return self.model

    def process_gradients(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50, norm_type=2)
