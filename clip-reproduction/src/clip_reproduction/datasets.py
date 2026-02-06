import torch
import torchvision
from torchvision import transforms

# Dataset registry
DATASETS = {
    "cifar100": torchvision.datasets.CIFAR100,
}


def get_dataset(
    name: str,
    root: str = "./data/raw",
    train: bool = True,
    download: bool = True,
    transform=None,
):
    if transform is None:
        transform = transforms.ToTensor()

    return DATASETS[name](
        root=root,
        train=train,
        transform=transform,
        download=download,
    )
