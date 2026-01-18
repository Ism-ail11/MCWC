from __future__ import annotations

from typing import Tuple


def load_cifar100(root: str = "./data", train: bool = False):
    from torchvision import datasets, transforms  # type: ignore

    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    return datasets.CIFAR100(root=root, train=train, download=True, transform=tfm)
