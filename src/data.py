import json
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NpyPuzzleDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        split_dir = os.path.join(data_dir, split)
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")

        with open(os.path.join(split_dir, "dataset.json"), "r") as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.inputs[idx])).long(),
            torch.from_numpy(np.array(self.labels[idx])).long(),
        )


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[NpyPuzzleDataset, DataLoader, DataLoader]:
    train_dataset = NpyPuzzleDataset(data_dir, split="train")
    val_split = "test" if os.path.exists(os.path.join(data_dir, "test")) else "train"
    val_dataset = NpyPuzzleDataset(data_dir, split=val_split)

    use_persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=use_persistent_workers,
    )

    return train_dataset, train_loader, val_loader
