import glob
import json
from os.path import join, exists
from pathlib import Path

import submitit
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from lipsim.core import utils
from lipsim.core.data.embedding_dataset import EmbeddingDataset
from lipsim.core.data.readers import readers_config
from lipsim.core.models.l2_lip.model import NormalizedModel, ClassificationLayer
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
        self.is_master = bool(self.rank == 0)
        self.ngpus = torch.cuda.device_count()
        self.world_size = self.num_nodes * self.ngpus
        self.embed_dim = N_CLASSES[self.config.teacher_model_name]

        means = (0.0000, 0.0000, 0.0000)
        stds = (1.0000, 1.0000, 1.0000)
        model = L2LipschitzNetworkV2(self.config, self.embed_dim)
        self.model = NormalizedModel(model, means, stds)
        self.model = self.model.cuda()
        utils.setup_distributed_training(self.world_size, self.rank, self.config.dist_url)
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.model = self.load_ckpt()
        self.model = self.model.eval()

        self.linear_classifier = ClassificationLayer(self.config,
                                                     embed_dim=N_CLASSES[self.config.teacher_model_name],
                                                     n_classes=1000)
        self.linear_classifier = self.linear_classifier.cuda()

        self.linear_classifier = DistributedDataParallel(
            self.linear_classifier, device_ids=[self.local_rank], output_device=self.local_rank)

        self.optimizer = utils.get_optimizer(self.config, self.model.parameters())

        train_dataset = EmbeddingDataset(root=self.config.data_dir, split='train')
        val_dataset = EmbeddingDataset(root=self.config.data_dir, split='val')
        self.sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, num_workers=4,
                                       shuffle=False,
                                       pin_memory=False, sampler=self.sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=4, shuffle=False,
                                     pin_memory=False)
        self.metric_logger = utils.MetricLogger(delimiter="  ")

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
        self.model.load_state_dict(new_checkpoint)
        return self.model

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
            torch.save(state, ckpt_path)

    # @record
    def __call__(self):
        self._init_class_properties()
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(self.config)).items())))
        cudnn.benchmark = True
        self.saved_ckpts = set([0])
        best_acc = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.epochs, eta_min=0)
        print(self.config.epochs)
        for epoch in range(0, self.config.epochs):
            self.sampler.set_epoch(epoch)
            train_stats = self.train(epoch)
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if epoch % self.config.frequency_log_steps == 0 or epoch == self.config.epochs - 1:
                test_stats = self.evaluate()
                print(
                    f"Accuracy at epoch {epoch} of the network on the test images: {test_stats['acc1']:.1f}%")
                best_acc = max(best_acc, test_stats["acc1"])
                print(f'Max accuracy so far: {best_acc:.2f}%')
                log_stats = {**{k: v for k, v in log_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()}}

            with (Path(self.train_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            self._save_ckpt(step=1, epoch=epoch)
        print("Training of the supervised linear classifier on frozen features completed.\n"
              "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
        self._save_ckpt(step=1, epoch=self.config.epochs, final=True)

    def train(self, epoch):
        self.linear_classifier.train()
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        for (inp, target) in self.metric_logger.log_every(self.train_loader, 20, header):
            # move to gpu
            inp = inp[:, 0, :, :, :].squeeze(1)
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = self.model(inp)

            output = self.linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(output, target)

            # compute the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # step
            self.optimizer.step()

            # log
            torch.cuda.synchronize()
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        self.metric_logger.synchronize_between_processes()
        print("Averaged stats:", self.metric_logger)
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self):
        self.linear_classifier.eval()

        for inp, target in self.val_loader:
            # self.val_sampler.set_epoch(0)
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = self.model(inp)
            output = self.linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)

            # if self.linear_classifier.module.num_labels >= 5:
            #     acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # else:
            acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            # if self.linear_classifier.module.num_labels >= 5:
            #     self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # if self.linear_classifier.module.num_labels >= 5:
        #     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #           .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5, losses=self.metric_logger.loss))
        # else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=self.metric_logger.acc1, losses=self.metric_logger.loss))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
