import random

import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from clip_reproduction.models.text import ByteTokenizer

DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "mnist": torchvision.datasets.MNIST,
    "fashionmnist": torchvision.datasets.FashionMNIST,
    "kmnist": torchvision.datasets.KMNIST,
}

PROMPT_TEMPLATES = {
    "cifar10": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "cifar100": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "fashionmnist": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "kmnist": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "mnist": [
        "a photo of the digit {label}",
        "a handwritten digit {label}",
        "an image of the number {label}",
    ],
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)


class ImageTextPairDataset(Dataset):
    def __init__(
        self,
        subset: Subset,
        class_names: list[str],
        tokenizer: ByteTokenizer,
        templates: list[str],
    ) -> None:
        self.subset = subset
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.templates = templates

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.subset[idx]
        class_name = self.class_names[label]
        prompt = random.choice(self.templates).format(label=class_name)
        tokens = self.tokenizer.encode(prompt)
        return image, tokens


def _build_transform(name: str, image_size: int, is_train: bool):
    if name == "cifar10":
        transforms_list = [transforms.Resize((image_size, image_size))]
        if is_train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ],
        )
        return transforms.Compose(transforms_list)

    if name == "cifar100":
        transforms_list = [transforms.Resize((image_size, image_size))]
        if is_train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ],
        )
        return transforms.Compose(transforms_list)

    if name == "mnist":
        transforms_list = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ]
        return transforms.Compose(transforms_list)
    
    if name == "fashionmnist":
        transforms_list = [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ]
        return transforms.Compose(transforms_list)

    raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASETS.keys())}")


def _split_indices(total_size: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total_size))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    indices = [indices[i] for i in permutation]
    val_size = int(total_size * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def get_classification_datasets(
    name: str,
    root: str,
    image_size: int,
    val_ratio: float,
    seed: int,
    download: bool = True,
) -> tuple[Subset, Subset, int, list[str]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASETS.keys())}")

    dataset_class = DATASETS[name]
    train_tf = _build_transform(name=name, image_size=image_size, is_train=True)
    eval_tf = _build_transform(name=name, image_size=image_size, is_train=False)

    train_base = dataset_class(root=root, train=True, transform=train_tf, download=download)
    eval_base = dataset_class(root=root, train=True, transform=eval_tf, download=download)

    if name == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = list(train_base.classes)
    num_classes = len(class_names)

    train_indices, val_indices = _split_indices(len(train_base), val_ratio=val_ratio, seed=seed)
    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(eval_base, val_indices)

    return train_subset, val_subset, num_classes, class_names


def get_classification_train_test_datasets(
    name: str,
    root: str,
    image_size: int,
    download: bool = True,
    train_transform=None,
    eval_transform=None,
) -> tuple[Dataset, Dataset, int, list[str]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASETS.keys())}")

    dataset_class = DATASETS[name]
    train_tf = train_transform if train_transform is not None else _build_transform(name=name, image_size=image_size, is_train=True)
    eval_tf = eval_transform if eval_transform is not None else _build_transform(name=name, image_size=image_size, is_train=False)

    train_ds = dataset_class(root=root, train=True, transform=train_tf, download=download)
    test_ds = dataset_class(root=root, train=False, transform=eval_tf, download=download)

    if name == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = list(train_ds.classes)
    num_classes = len(class_names)

    return train_ds, test_ds, num_classes, class_names


def get_clip_datasets(
    name: str,
    root: str,
    image_size: int,
    val_ratio: float,
    seed: int,
    tokenizer: ByteTokenizer,
    download: bool = True,
) -> tuple[ImageTextPairDataset, ImageTextPairDataset, int, list[str]]:
    train_subset, val_subset, num_classes, class_names = get_classification_datasets(
        name=name,
        root=root,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        download=download,
    )

    train_dataset = ImageTextPairDataset(
        subset=train_subset,
        class_names=class_names,
        tokenizer=tokenizer,
        templates=PROMPT_TEMPLATES[name],
    )
    val_dataset = ImageTextPairDataset(
        subset=val_subset,
        class_names=class_names,
        tokenizer=tokenizer,
        templates=PROMPT_TEMPLATES[name],
    )

    return train_dataset, val_dataset, num_classes, class_names
