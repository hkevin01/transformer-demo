from __future__ import annotations

import argparse

from src.evaluate import EvalConfig, evaluate_models
from src.train import TrainConfig, train_models
from src.visualize import generate_attention_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer self-attention demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train transformer and baseline models")
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--seq-len", type=int, default=16)
    train_parser.add_argument("--vocab-size", type=int, default=20)
    train_parser.add_argument("--embed-dim", type=int, default=64)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--ff-hidden-dim", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--train-size", type=int, default=4000)
    train_parser.add_argument("--val-size", type=int, default=800)
    train_parser.add_argument("--test-size", type=int, default=800)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--artifacts-dir", type=str, default="artifacts")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--seq-len", type=int, default=16)
    eval_parser.add_argument("--vocab-size", type=int, default=20)
    eval_parser.add_argument("--embed-dim", type=int, default=64)
    eval_parser.add_argument("--num-heads", type=int, default=4)
    eval_parser.add_argument("--ff-hidden-dim", type=int, default=128)
    eval_parser.add_argument("--num-layers", type=int, default=2)
    eval_parser.add_argument("--dropout", type=float, default=0.1)
    eval_parser.add_argument("--test-size", type=int, default=800)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--artifacts-dir", type=str, default="artifacts")

    viz_parser = subparsers.add_parser("visualize", help="Generate attention and comparison plots")
    viz_parser.add_argument("--sample-index", type=int, default=0)
    viz_parser.add_argument("--seq-len", type=int, default=16)
    viz_parser.add_argument("--vocab-size", type=int, default=20)
    viz_parser.add_argument("--embed-dim", type=int, default=64)
    viz_parser.add_argument("--num-heads", type=int, default=4)
    viz_parser.add_argument("--ff-hidden-dim", type=int, default=128)
    viz_parser.add_argument("--num-layers", type=int, default=2)
    viz_parser.add_argument("--dropout", type=float, default=0.1)
    viz_parser.add_argument("--seed", type=int, default=42)
    viz_parser.add_argument("--artifacts-dir", type=str, default="artifacts")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_hidden_dim=args.ff_hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
        )
        history = train_models(cfg)
        last_t = history["transformer"]["val_acc"][-1]
        last_b = history["baseline"]["val_acc"][-1]
        print(f"Training complete: transformer val_acc={last_t:.3f}, baseline val_acc={last_b:.3f}")
        return

    if args.command == "evaluate":
        cfg = EvalConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_hidden_dim=args.ff_hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            test_size=args.test_size,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
        )
        metrics = evaluate_models(cfg)
        print(metrics)
        return

    if args.command == "visualize":
        generate_attention_visualizations(
            artifacts_dir=args.artifacts_dir,
            sample_index=args.sample_index,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_hidden_dim=args.ff_hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seed=args.seed,
        )
        print("Visualization complete. See artifacts directory.")
        return


if __name__ == "__main__":
    main()
