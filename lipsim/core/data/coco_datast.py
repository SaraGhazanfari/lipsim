import os.path
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection


class COCODataset(CocoDetection):

    def __init__(
            self, config: Optional, root: str, batch_size: int,
            annFile='path2json', num_workers=10,
            transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            is_training=True, is_distributed=False, world_size=1
    ) -> None:
        super().__init__(root=config.data_dir, transforms=transforms, transform=transform, annFile=annFile)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)

        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def __len__(self) -> int:
        return len(self.ids)

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
