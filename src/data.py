from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetConfig:
    vocab_size: int = 20
    seq_len: int = 16
    train_size: int = 4000
    val_size: int = 800
    test_size: int = 800
    batch_size: int = 64
    num_workers: int = 0


class EndpointEqualityDataset(Dataset):
    """Synthetic dataset: label is 1 if first token equals last token."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


def build_endpoint_equality_split(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(num_samples, seq_len),
        generator=generator,
        dtype=torch.long,
    )
    labels = (inputs[:, 0] == inputs[:, -1]).long()
    return inputs, labels


def build_dataloaders(
    config: DatasetConfig,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    g_train = torch.Generator().manual_seed(seed)
    g_val = torch.Generator().manual_seed(seed + 1)
    g_test = torch.Generator().manual_seed(seed + 2)

    train_x, train_y = build_endpoint_equality_split(
        num_samples=config.train_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        generator=g_train,
    )
    val_x, val_y = build_endpoint_equality_split(
        num_samples=config.val_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        generator=g_val,
    )
    test_x, test_y = build_endpoint_equality_split(
        num_samples=config.test_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        generator=g_test,
    )

    train_ds = EndpointEqualityDataset(train_x, train_y)
    val_ds = EndpointEqualityDataset(val_x, val_y)
    test_ds = EndpointEqualityDataset(test_x, test_y)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }
    return loaders
