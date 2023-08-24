import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from lipsim.core.utils import get_preprocess_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class ImagenetDataset(Dataset):
    def __init__(self, config, batch_size, is_training, is_distributed=False, num_workers=8):
        self.config = config
        self.batch_size = batch_size
        self.is_training = is_training
        self.is_distributed = is_distributed
        self.num_workers = num_workers
        self.n_classes = 768
        self.height, self.width = 224, 500
        self.n_train_files = 1_281_167
        self.n_test_files = 50_1000
        self.img_size = (None, 3, 224, 500)

        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)

        self.samples = []
        self.targets = []
        self.transform = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }
        self.syn_to_class = {}
        self.split = 'train' if self.is_training else 'val'
        self._get_samples()

    def _get_samples(self):

        with open(os.path.join(self.config.data_dir, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(self.config.data_dir, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(self.config.data_dir, "ILSVRC/Data/CLS-LOC", self.split)
        for entry in os.listdir(samples_dir):
            if self.split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif self.split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform[self.split](x)
        return x, self.targets[idx]

    def load_dataset(self):
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(self, shuffle=self.is_training)
        dataloader = DataLoader(
            self,
            batch_size=self.batch_size,  # may need to reduce this depending on your GPU
            num_workers=self.num_workers,  # may need to reduce this depending on your num of CPUs and RAM
            shuffle=self.is_training,
            drop_last=False,
            pin_memory=True,
            sampler=sampler
        )
        return dataloader, sampler


class NightDataset(Dataset):
    def __init__(self, config, batch_size, is_training=True, is_distributed=False, num_workers=4, split: str = "test_imagenet",
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
        self.n_classes = 768
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

