from datadings.torch import CompressedToPIL
from torchvision.transforms import transforms
from datadings.torch import Compose


def yfcc_transform(train=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    if train:
        crop = transforms.RandomCrop(224)
    else:
        crop = transforms.CenterCrop(224)

    t = {'image': Compose(
        CompressedToPIL(),
        transforms.Resize(256),
        crop,
        transforms.ToTensor(),
        normalize
    )}
    return t
