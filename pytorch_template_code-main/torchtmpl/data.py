# coding: utf-8

# Standard imports
import logging
import random

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms

import PlanktonDataset
import matplotlib.pyplot as plt


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    base_dataset = PlanktonDataset.PlanktonDataset(
        dir=data_config["trainpath"],
        train=True,
        patch_size=256,
        transform=input_transform,
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = 2
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes

def get_test_dataloaders(data_config, use_cuda):
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    base_dataset = PlanktonDataset.PlanktonDataset(
        dir=data_config["testpath"],
        train=False,
        patch_size=256,
        transform=input_transform,
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    # Build the dataloaders
    test_loader = torch.utils.data.DataLoader(
        base_dataset,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = 2
    input_size = tuple(base_dataset[0][0].shape)

    return test_loader, input_size, num_classes

