import logging
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageNet
from os.path import basename, join
from PIL import Image

def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, "rb") as f:
    img = Image.open(f)
    return img.convert("RGB")


class ImageNetEmbeddingDataset(ImageNet):

  def __init__(self, root, root_embedding, split, transform):
    super(ImageNetEmbeddingDataset, self).__init__(
      root, split=split, transform=transform)
    self.root_embedding = join(root_embedding, split)

    self.samples_embedding = []
    for path, _ in self.samples:
      dirname = path.split('/')[-2]
      filename = basename(path).split('.')[0]
      self.samples_embedding.append(
        join(self.root_embedding, dirname, f'{filename}.pkl')
      )

  def __getitem__(self, index: int):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path, _ = self.samples[index]
    path_embedding = self.samples_embedding[index]
    sample = pil_loader(path)
    sample_embedding = torch.load(path_embedding)
    if self.transform is not None:
      sample = self.transform(sample)
    return sample, sample_embedding

  def __len__(self) -> int:
      return len(self.samples)





