from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from tqdm.auto import tqdm

from src.data import DatasetConfig, build_dataloaders
from src.model.transformer_model import BaselineSequenceClassifier, TransformerSequenceClassifier
from src.utils import ensure_dir, get_device, save_json, set_seed


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    embed_dim: int = 64
    num_heads: int = 4
    ff_hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    artifacts_dir: str = "artifacts"
    vocab_size: int = 20
    seq_len: int = 16
    train_size: int = 4000
    val_size: int = 800
    test_size: int = 800
    batch_size: int = 64


def _run_epoch_transformer(
    model: TransformerSequenceClassifier,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)
    for tokens, labels in pbar:
        tokens = tokens.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits, _ = model(tokens)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        total_loss += loss.item() * tokens.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += tokens.size(0)

    return total_loss / total_samples, total_correct / total_samples


def _run_epoch_baseline(
    model: BaselineSequenceClassifier,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)
    for tokens, labels in pbar:
        tokens = tokens.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(tokens)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        total_loss += loss.item() * tokens.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += tokens.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_models(config: TrainConfig) -> Dict[str, Dict[str, list]]:
    set_seed(config.seed)
    device = get_device()

    data_config = DatasetConfig(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
    )
    loaders = build_dataloaders(data_config, seed=config.seed)

    transformer = TransformerSequenceClassifier(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_hidden_dim=config.ff_hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    baseline = BaselineSequenceClassifier(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        embed_dim=config.embed_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    transformer_opt = torch.optim.AdamW(
        transformer.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    baseline_opt = torch.optim.AdamW(
        baseline.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = {
        "transformer": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
        "baseline": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
        "config": asdict(config),
    }

    for epoch in range(1, config.epochs + 1):
        t_train_loss, t_train_acc = _run_epoch_transformer(
            model=transformer,
            loader=loaders["train"],
            optimizer=transformer_opt,
            criterion=criterion,
            device=device,
            train=True,
        )
        t_val_loss, t_val_acc = _run_epoch_transformer(
            model=transformer,
            loader=loaders["val"],
            optimizer=transformer_opt,
            criterion=criterion,
            device=device,
            train=False,
        )

        b_train_loss, b_train_acc = _run_epoch_baseline(
            model=baseline,
            loader=loaders["train"],
            optimizer=baseline_opt,
            criterion=criterion,
            device=device,
            train=True,
        )
        b_val_loss, b_val_acc = _run_epoch_baseline(
            model=baseline,
            loader=loaders["val"],
            optimizer=baseline_opt,
            criterion=criterion,
            device=device,
            train=False,
        )

        history["transformer"]["train_loss"].append(t_train_loss)
        history["transformer"]["val_loss"].append(t_val_loss)
        history["transformer"]["train_acc"].append(t_train_acc)
        history["transformer"]["val_acc"].append(t_val_acc)

        history["baseline"]["train_loss"].append(b_train_loss)
        history["baseline"]["val_loss"].append(b_val_loss)
        history["baseline"]["train_acc"].append(b_train_acc)
        history["baseline"]["val_acc"].append(b_val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"T(val_acc)={t_val_acc:.3f} "
            f"B(val_acc)={b_val_acc:.3f}"
        )

    artifacts_dir = ensure_dir(config.artifacts_dir)
    torch.save(transformer.state_dict(), artifacts_dir / "transformer.pt")
    torch.save(baseline.state_dict(), artifacts_dir / "baseline.pt")
    save_json(artifacts_dir / "metrics.json", history)
    return history


def load_models_for_eval(
    artifacts_dir: str | Path,
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    ff_hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> Tuple[TransformerSequenceClassifier, BaselineSequenceClassifier, torch.device]:
    device = get_device()
    artifacts_path = Path(artifacts_dir)

    transformer = TransformerSequenceClassifier(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    baseline = BaselineSequenceClassifier(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)

    transformer.load_state_dict(torch.load(artifacts_path / "transformer.pt", map_location=device))
    baseline.load_state_dict(torch.load(artifacts_path / "baseline.pt", map_location=device))

    transformer.eval()
    baseline.eval()
    return transformer, baseline, device
