import os

import submitit
import torch
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from lipsim.core import utils
import torch.distributed as dist


class ReturnIndexDataset(ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


class KNNEval:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self._setup_distributed_run()
        print('Reading data...')
        self._load_dataloader()
        print('Loading Features...')
        self._load_features()

    def _setup_distributed_run(self):
        cudnn.benchmark = True
        # self.train_dir = self.config.train_dir
        # self.ngpus = torch.cuda.device_count()
        # job_env = submitit.JobEnvironment()
        # self.rank = int(job_env.global_rank)
        # self.local_rank = int(job_env.local_rank)
        # self.num_nodes = int(job_env.num_nodes)
        # self.num_tasks = int(job_env.num_tasks)
        # self.is_master = bool(self.rank == 0)
        # torch.cuda.init()
        # self.world_size = 1
        # self.is_distributed = False
        # if self.num_nodes > 1 or self.num_tasks > 1:
        #     self.is_distributed = True
        #     self.world_size = self.num_nodes * self.ngpus
        # torch.cuda.set_device(self.local_rank)
        self.rank, self.world_size, self.local_rank, self.is_distributed = 0, 0, 0, False
        if self.is_distributed:
            utils.setup_distributed_training(self.world_size, self.rank)
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        dist.init_process_group(
            backend="nccl",
            init_method='127.0.0.1:29500',
            world_size=self.world_size,
            rank=self.rank,
        )
        torch.cuda.set_device(self.local_rank)

    def _load_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset_train = ReturnIndexDataset(os.path.join(self.config.data_dir, "train"), transform=transform)
        self.sampler = None  # torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
        dataset_val = ReturnIndexDataset(os.path.join(self.config.data_dir, "val"), transform=transform)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config.batch_size,
            num_workers=1,
            sampler=self.sampler,
            pin_memory=True,
            drop_last=False,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.config.batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )

    def _load_features(self):
        # if self.config.load_features:
        #     self.train_features = torch.load(os.path.join(self.config.load_features, "trainfeat.pth"))
        #     self.test_features = torch.load(os.path.join(self.config.load_features, "testfeat.pth"))
        #     self.train_labels = torch.load(os.path.join(self.config.load_features, "trainlabels.pth"))
        #     self.test_labels = torch.load(os.path.join(self.config.load_features, "testlabels.pth"))
        # else:
        self.train_features, self.test_features, self.train_labels, self.test_labels = self._extract_feature_pipeline()
        self.train_features = self.train_features.cuda()
        self.test_features = self.test_features.cuda()
        self.train_labels = self.train_labels.cuda()
        self.test_labels = self.test_labels.cuda()

    def _extract_feature_pipeline(self):

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features = self.extract_features(self.train_loader)
        print("Extracting features for val set...")
        test_features = self.extract_features(self.test_loader)

        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

        train_labels = torch.tensor([s[-1] for s in self.train_loader.dataset.samples]).long()
        test_labels = torch.tensor([s[-1] for s in self.test_loader.dataset.samples]).long()
        if dist.get_rank() == 0:
            torch.save(train_features.cpu(), os.path.join(self.config.dump_features, "trainfeat.pth"))
            torch.save(test_features.cpu(), os.path.join(self.config.dump_features, "testfeat.pth"))
            torch.save(train_labels.cpu(), os.path.join(self.config.dump_features, "trainlabels.pth"))
            torch.save(test_labels.cpu(), os.path.join(self.config.dump_features, "testlabels.pth"))
        return train_features, test_features, train_labels, test_labels

    def extract_features(self, data_loader):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        for samples, index in metric_logger.log_every(data_loader, 10):
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            feats = self.model(samples).clone()

            if dist.get_rank() == 0 and features is None:
                features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
                features = features.cuda(non_blocking=True)
                print(f"Storing features into tensor of shape {features.shape}")

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            # update storage feature matrix
            if dist.get_rank() == 0:
                features.index_copy_(0, index_all, torch.cat(output_l))
        return features

    def knn_classifier(self):
        print("Features are ready!\nStart the k-NN classification.")
        for k in self.config.nb_knn:
            top1, top5 = self.knn_classifier_for_each_k(k, self.config.temperature)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

        dist.barrier()

    @torch.no_grad()
    def knn_classifier_for_each_k(self, k, T, num_classes=1000):

        top1, top5, total = 0.0, 0.0, 0
        train_features = self.train_features.t()
        num_test_images, num_chunks = self.test_labels.shape[0], 100
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = self.test_features[
                       idx: min((idx + imgs_per_chunk), num_test_images), :
                       ]
            targets = self.test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = self.train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(T).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
            total += targets.size(0)
        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        return top1, top5
