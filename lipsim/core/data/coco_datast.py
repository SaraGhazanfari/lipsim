from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


class COCODataset:

    def __init__(
            self, config: Optional, batch_size: int, num_workers=10,
            is_training=True, is_distributed=False, world_size=1) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_dataloader(self):
        dataset = datasets.ImageFolder(self.config.data_dir, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False), None
