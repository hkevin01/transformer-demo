from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from src.data import DatasetConfig, build_dataloaders
from src.train import load_models_for_eval


@dataclass
class EvalConfig:
    artifacts_dir: str = "artifacts"
    vocab_size: int = 20
    seq_len: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    ff_hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    test_size: int = 800
    batch_size: int = 64
    seed: int = 42


def evaluate_models(config: EvalConfig) -> Dict[str, float]:
    data_config = DatasetConfig(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        train_size=1,
        val_size=1,
        test_size=config.test_size,
        batch_size=config.batch_size,
    )
    loaders = build_dataloaders(data_config, seed=config.seed)
    test_loader = loaders["test"]

    transformer, baseline, device = load_models_for_eval(
        artifacts_dir=config.artifacts_dir,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_hidden_dim=config.ff_hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    transformer_correct = 0
    baseline_correct = 0
    total = 0

    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            t_logits, _ = transformer(tokens)
            b_logits = baseline(tokens)

            transformer_correct += (t_logits.argmax(dim=-1) == labels).sum().item()
            baseline_correct += (b_logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    metrics = {
        "transformer_test_acc": transformer_correct / total,
        "baseline_test_acc": baseline_correct / total,
    }
    return metrics
