import glob
import os
import json
from os.path import join
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from lipsim.core import utils
from lipsim.core.data.readers import N_CLASSES, readers_config
from lipsim.core.models.l2_lip.model import L2LipschitzNetwork, NormalizedModel, ClassificationLayer


class LinearEvaluation:
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.embed_dim = N_CLASSES[self.config.teacher_model_name]
        self.model = self.load_ckpt()
        self.model = self.model.eval()
        self.linear_classifier = ClassificationLayer(self.config,
                                                     embed_dim=N_CLASSES[self.config.teacher_model_name],
                                                     n_classes=1000)
        self.linear_classifier = self.linear_classifier.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.linear_classifier.parameters(),
            self.config.lr * self.config.batch_size,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        Reader = readers_config[self.config.dataset]
        self.train_loader, _ = Reader(config=self.config, batch_size=self.config.batch_size,
                                      is_training=True).load_dataset()
        self.val_loader, _ = Reader(config=self.config, batch_size=self.config.batch_size,
                                    is_training=False).load_dataset()
        self.metric_logger = utils.MetricLogger(delimiter="  ")

    def load_ckpt(self):
        means = (0.0000, 0.0000, 0.0000)
        stds = (1.0000, 1.0000, 1.0000)
        model = L2LipschitzNetwork(self.config, self.embed_dim)
        model = NormalizedModel(model, means, stds)
        model = torch.nn.DataParallel(model)
        model = model.to(self.device)
        checkpoints = glob.glob(join(self.config.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
        print(checkpoints)
        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
        print(ckpt_name)
        ckpt_path = join(self.config.train_dir, 'checkpoints', ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'alpha' not in k:
                new_checkpoint[k] = v
        model.load_state_dict(new_checkpoint)
        return model

    def eval_linear(self):
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(self.config)).items())))
        cudnn.benchmark = True

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.epochs, eta_min=0)
        print(self.config.epochs)
        for epoch in range(0, self.config.epochs):
            train_stats = self.train(epoch)
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if epoch % self.config.val_freq == 0 or epoch == self.config.epochs - 1:
                test_stats = self.evaluate()
                print(
                    f"Accuracy at epoch {epoch} of the network on the test images: {test_stats['acc1']:.1f}%")
                best_acc = max(best_acc, test_stats["acc1"])
                print(f'Max accuracy so far: {best_acc:.2f}%')
                log_stats = {**{k: v for k, v in log_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()}}

            with (Path(self.config.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": self.linear_classifier.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(self.config.output_dir, "checkpoint.pth.tar"))
        print("Training of the supervised linear classifier on frozen features completed.\n"
              "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))

    def train(self, epoch):
        self.linear_classifier.train()
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        for (inp, target) in self.metric_logger.log_every(self.train_loader, 20, header):
            # move to gpu
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
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = self.model(inp)
            output = self.linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)

            if self.linear_classifier.module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            self.metric_logger.update(loss=loss.item())
            self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            if self.linear_classifier.module.num_labels >= 5:
                self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if self.linear_classifier.module.num_labels >= 5:
            print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5, losses=self.metric_logger.loss))
        else:
            print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=self.metric_logger.acc1, losses=self.metric_logger.loss))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
