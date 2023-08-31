import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from lipsim.core.utils import get_preprocess_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lipsim.core.utils import GaussianBlur, Solarization


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.standard_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        images = []
        images.append(self.standard_transform(image))
        images.append(self.global_transfo1(image))
        images.append(self.global_transfo2(image))
        images = torch.stack(images, dim=0)
        return images


class ImagenetDataset(Dataset):
    def __init__(self, config, batch_size, is_training, is_distributed=False, num_workers=0, world_size=1):
        self.config = config
        self.batch_size = batch_size
        self.is_training = is_training
        self.is_distributed = is_distributed
        self.world_size = world_size
        self.num_workers = num_workers
        self.n_classes = 768
        self.height, self.width = 224, 500
        self.n_train_files = 1_281_167
        self.n_test_files = 50_1000
        self.img_size = (None, 3, 224, 500)
        self.split = 'train' if self.is_training else 'val'

        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)

        self.samples = []
        self.targets = []
        self.transform = {
            # 'train': transforms.Compose([
            #     transforms.CenterCrop(224),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ]),
            'train': DataAugmentationDINO(
                global_crops_scale=(0.4, 1.),
                local_crops_scale=(0.05, 0.4),
                local_crops_number=8
            ),
            'val': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }

    def load_dataset(self):
        sampler = None
        shuffle = True if self.is_training else False
        dataset = datasets.ImageFolder(self.config.data_dir, transform=self.transform[self.split])
        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=False, num_replicas=self.world_size)

        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, shuffle=shuffle,
                                 num_workers=self.num_workers, pin_memory=True, drop_last=True)
        return data_loader, sampler


class NightDataset(Dataset):
    def __init__(self, config, batch_size, is_training=True, is_distributed=False, num_workers=4,
                 split: str = "test_imagenet",
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 preprocess: str = "DEFAULT", **kwargs):
        self.root_dir = config.data_dir
        self.num_workers = num_workers
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6]  # Filter out triplets with less than 6 unanimous votes
        self.split = split
        self.batch_size = batch_size
        self.interpolation = interpolation
        self.preprocess_fn = get_preprocess_fn(preprocess, 224, self.interpolation)
        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)
        self.n_classes = N_CLASSES[config.teacher_model_name]
        if self.split == "train" or self.split == "val":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        img_ref = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4])))
        img_left = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5])))
        img_right = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6])))
        return img_ref, img_left, img_right, p, id

    def load_dataset(self):
        return DataLoader(self, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False), len(self.csv)


readers_config = {
    'imagenet-1k': ImagenetDataset,
    'night': NightDataset,
}

N_CLASSES = {
    'dino_vitb16': 768,
    'open_clip_vitb32': 512,
    'clip_vitb32': 512,
}
