from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.data import DatasetConfig, build_dataloaders
from src.train import load_models_for_eval
from src.utils import ensure_dir, load_json


def plot_attention_heatmaps(
    attention_maps: List[torch.Tensor],
    tokens: torch.Tensor,
    out_dir: str | Path,
) -> None:
    out_path = ensure_dir(out_dir)
    token_labels = [str(int(t)) for t in tokens.tolist()]

    for layer_idx, attn in enumerate(attention_maps):
        # attn shape: (batch, heads, target_len, source_len)
        attn_first = attn[0].detach().cpu()
        num_heads = attn_first.size(0)

        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4), squeeze=False)
        for head_idx in range(num_heads):
            ax = axes[0, head_idx]
            sns.heatmap(
                attn_first[head_idx],
                ax=ax,
                cmap="viridis",
                xticklabels=token_labels,
                yticklabels=token_labels,
                cbar=True,
            )
            ax.set_title(f"Layer {layer_idx + 1} Head {head_idx + 1}")
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")

        fig.tight_layout()
        fig.savefig(out_path / f"attention_heatmap_layer{layer_idx + 1}.png", dpi=140)
        plt.close(fig)


def plot_training_comparison(metrics_json_path: str | Path, out_dir: str | Path) -> None:
    history: Dict = load_json(metrics_json_path)
    out_path = ensure_dir(out_dir)

    epochs = range(1, len(history["transformer"]["val_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["transformer"]["val_acc"], label="Transformer")
    axes[0].plot(epochs, history["baseline"]["val_acc"], label="Baseline (No Attention)")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["transformer"]["val_loss"], label="Transformer")
    axes[1].plot(epochs, history["baseline"]["val_loss"], label="Baseline (No Attention)")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path / "training_comparison.png", dpi=140)
    plt.close(fig)


def generate_attention_visualizations(
    artifacts_dir: str = "artifacts",
    sample_index: int = 0,
    vocab_size: int = 20,
    seq_len: int = 16,
    embed_dim: int = 64,
    num_heads: int = 4,
    ff_hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    seed: int = 42,
) -> None:
    transformer, _, device = load_models_for_eval(
        artifacts_dir=artifacts_dir,
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    data_config = DatasetConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        train_size=1,
        val_size=1,
        test_size=max(sample_index + 1, 8),
        batch_size=max(sample_index + 1, 8),
    )
    test_loader = build_dataloaders(data_config, seed=seed)["test"]
    tokens_batch, _ = next(iter(test_loader))

    sample_tokens = tokens_batch[sample_index : sample_index + 1].to(device)
    with torch.no_grad():
        _, attention_maps = transformer(sample_tokens)

    plot_attention_heatmaps(
        attention_maps=attention_maps,
        tokens=sample_tokens[0].detach().cpu(),
        out_dir=artifacts_dir,
    )
    plot_training_comparison(Path(artifacts_dir) / "metrics.json", artifacts_dir)
