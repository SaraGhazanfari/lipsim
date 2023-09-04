import os.path
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.utils.data
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy', ]


def is_image_file(filename, mode='img'):
    if mode == 'img':
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif mode == 'np':
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


# from IPython import embed
def make_dataset(dir, mode='img'):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, mode=mode):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class BAPPSDataset(Dataset):
    def __init__(self, data_root, load_size=64, split='val', dataset='cnn'):
        self.is_training = True if split == 'train' else False
        self.root = os.path.join(data_root, split, dataset)
        self.load_size = load_size

        # image directory
        self.dir_ref = os.path.join(self.root, 'ref')
        self.ref_paths = make_dataset(self.dir_ref)
        self.ref_paths = sorted(self.ref_paths)

        self.dir_p0 = os.path.join(self.root, 'p0')
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = os.path.join(self.root, 'p1')
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        # transform_list = [transforms.Resize(load_size)]
        pad_size = int((load_size - 64)/2)
        transform_list = [transforms.Pad([pad_size, pad_size])]
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.dir_J = os.path.join(self.root, 'judge')
        self.judge_paths = make_dataset(self.dir_J, mode='np')
        self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = Image.open(p0_path).convert('RGB')
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = Image.open(p1_path).convert('RGB')
        p1_img = self.transform(p1_img_)

        ref_path = self.ref_paths[index]
        ref_img_ = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img_)

        judge_path = self.judge_paths[index]
        judge_img = np.load(judge_path).reshape((1, 1, 1,))  # [0,1]

        judge_img = torch.FloatTensor(judge_img)
        return ref_img, p0_img, p1_img, judge_img, index

    def __len__(self):
        return len(self.p0_paths)

    def get_dataloader(self, batch_size=1, num_workers=8):
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                                 shuffle=self.is_training,
                                                 num_workers=1)
        return dataloader
