import logging
from os.path import join, exists

import logging
from os.path import join, exists

import submitit
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from lipsim.core import utils
from lipsim.core.cosine_scheduler import CosineAnnealingWarmupRestarts
from lipsim.core.data.embedding_dataset import EmbeddingDataset
from lipsim.core.models.l2_lip.model import ClassificationLayer
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
        self.is_master = bool(self.rank == 0)
        self.ngpus = torch.cuda.device_count()
        self.world_size = self.num_nodes * self.ngpus
        self.embed_dim = N_CLASSES[self.config.teacher_model_name]

        self.message = utils.MessageBuilder()
        utils.setup_logging(self.config, 0)
        utils.setup_distributed_training(self.world_size, self.rank, self.config.dist_url)
        # means = (0.0000, 0.0000, 0.0000)
        # stds = (1.0000, 1.0000, 1.0000)
        # model = L2LipschitzNetworkV2(self.config, self.embed_dim)
        #
        # self.model = NormalizedModel(model, means, stds)
        # self.model = self.model.cuda()
        # self.model = DistributedDataParallel(
        #     self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        #
        # self.model = self.load_ckpt()
        # self.model = self.model.eval()

        self.linear_classifier = ClassificationLayer(self.config,
                                                     embed_dim=N_CLASSES[self.config.teacher_model_name],
                                                     n_classes=1000)
        self.linear_classifier = self.linear_classifier.cuda()

        self.linear_classifier = DistributedDataParallel(
            self.linear_classifier, device_ids=[self.local_rank], output_device=self.local_rank)

        self.optimizer = utils.get_optimizer(self.config, self.linear_classifier.parameters())

        self.train_dataset = EmbeddingDataset(root=self.config.data_dir, split='train')
        val_dataset = EmbeddingDataset(root=self.config.data_dir, split='val')
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=4,
                                       shuffle=False,
                                       pin_memory=False, sampler=self.sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=4, shuffle=False,
                                     pin_memory=False)

    # def load_ckpt(self):
    #     checkpoints = glob.glob(join(self.config.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
    #     get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
    #     ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
    #     ckpt_path = join(self.config.train_dir, 'checkpoints', ckpt_name)
    #     checkpoint = torch.load(ckpt_path)
    #     new_checkpoint = {}
    #     for k, v in checkpoint['model_state_dict'].items():
    #         if 'alpha' not in k:
    #             new_checkpoint[k] = v
    #     self.model.load_state_dict(new_checkpoint)
    #     return self.model

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
                'model_state_dict': self.linear_classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict()
            }
            torch.save(state, ckpt_path)

    # @record
    def __call__(self):
        self._init_class_properties()
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(self.config)).items())))
        cudnn.benchmark = True
        self.saved_ckpts = set([0])
        num_steps = (self.config.epochs * len(self.train_dataset) // (
                self.config.batch_size * torch.cuda.device_count()))
        self.scheduler = CosineAnnealingWarmupRestarts(optimizer=self.optimizer, max_lr=self.config.lr,
                                                       min_lr=0,
                                                       first_cycle_steps=num_steps,
                                                       warmup_steps=num_steps * 5 / self.config.epochs)
        for epoch in range(0, self.config.epochs):
            self.sampler.set_epoch(epoch)
            self.train(epoch)

            if epoch % self.config.frequency_log_steps == 0 or epoch == self.config.epochs - 1:
                acc1, loss = self.evaluate()
                self.message.add("epoch", epoch, format="4.2f")
                self.message.add("loss", loss, format=".4f")
                self.message.add("acc", acc1, format=".4f")
                logging.info(self.message.get_message())
            self._save_ckpt(step=1, epoch=epoch)
        self._save_ckpt(step=1, epoch=self.config.epochs, final=True)

    def train(self, epoch):
        self.linear_classifier.train()
        for idx, (inp, target) in tqdm(enumerate(self.train_loader)):
            self.optimizer.zero_grad()
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = self.linear_classifier(inp)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.synchronize()
            if idx % 1000 == 999:
                lr = self.optimizer.param_groups[0]['lr']
                self.message.add("epoch_id", epoch)
                self.message.add("epoch", idx / len(self.train_loader), format="4.2f")
                self.message.add("step", idx + 1, width=5, format=".0f")
                self.message.add("lr", lr, format=".6f")
                self.message.add("loss", loss, format=".4f")
                logging.info(self.message.get_message())

    @torch.no_grad()
    def evaluate(self):
        self.linear_classifier.eval()
        for inp, target in self.val_loader:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = self.linear_classifier(inp)
            loss = nn.CrossEntropyLoss()(output, target)
            acc1, = utils.accuracy(output, target, topk=(1,))
        return acc1, loss
