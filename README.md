<div align="center">

# Transformer Demo

### Self-Attention and Transformer Encoder Blocks - Built from Scratch in PyTorch

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/OWNER/REPO/ci.yml?label=CI&logo=github)](https://github.com/OWNER/REPO/actions)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![Maintained](https://img.shields.io/badge/Maintained-yes-22c55e)](https://github.com/OWNER/REPO)

</div>

---

## Overview

This project is a **fully runnable, from-scratch implementation** of the Transformer encoder architecture using PyTorch. It was built to make the internals of self-attention and the Transformer block concrete and inspectable rather than hiding them inside framework abstractions. Every component - scaled dot-product attention, multi-head self-attention, the position-wise feed-forward network, residual connections, and layer normalization - is implemented as a plain `nn.Module` and is readable in a few dozen lines of code.

The task chosen to demonstrate the model is **toy sequence classification**: given an integer token sequence of fixed length, the model must predict `1` if the first and last tokens are equal and `0` otherwise. This problem is deliberately designed to require long-range token interaction, which is exactly the strength of self-attention. A baseline MLP model that sees the same tokens but has no attention mechanism is trained alongside the Transformer to make the performance gap visible and measurable.

> [!NOTE]
> This is an educational project. The model sizes are intentionally small so that training completes in seconds on CPU. The architecture choices directly mirror the original "Attention Is All You Need" encoder design (Vaswani et al., 2017).

---

## Table of Contents

- [Why Self-Attention?](#why-self-attention)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Model Comparison](#model-comparison)
- [Notebook Exploration](#notebook-exploration)
- [API Reference](#api-reference)

---

## Why Self-Attention?

Traditional sequence models like RNNs and CNNs process tokens either one at a time or within a fixed local window. This makes it structurally difficult for them to relate tokens that are far apart in a sequence. Self-attention solves this by computing a **direct pairwise relationship score between every token and every other token** in the sequence simultaneously. The result is a weighted mixture of all token representations, where the weights reflect how relevant each other position is when computing the representation for the current position.

The scaling factor $\frac{1}{\sqrt{d_k}}$ in the attention formula prevents the dot products from growing large in high-dimensional spaces, which would push the softmax into regions with near-zero gradients. Multi-head attention repeats this process in parallel across several learned subspaces, allowing the model to jointly attend to information from different representational perspectives at different positions.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

> [!IMPORTANT]
> The task (first token equals last token) is designed so that a model **without** attention cannot reliably solve it after mean-pooling, because averaging destroys positional information. The Transformer can solve it precisely because attention can learn to directly compare position 0 and position $L-1$.

---

## Architecture Deep Dive

### Scaled Dot-Product Attention

The atomic unit of the Transformer is scaled dot-product attention. The query matrix $Q$, key matrix $K$, and value matrix $V$ are all linear projections of the input. Dot products between queries and keys produce a raw score matrix. After scaling and softmax normalization, these scores become attention weights that are used to compute a weighted sum of the value vectors. The result is a new representation for each token that incorporates context from the entire sequence.

```mermaid
flowchart TD
    X["Input X\n(batch, seq, embed_dim)"]
    Q["Q = X · W_Q\n(batch, heads, seq, head_dim)"]
    K["K = X · W_K\n(batch, heads, seq, head_dim)"]
    V["V = X · W_V\n(batch, heads, seq, head_dim)"]
    SCORES["Scores = QKᵀ / √d_k\n(batch, heads, seq, seq)"]
    MASK["Optional Mask\n(additive or boolean)"]
    SOFTMAX["Softmax → Attention Weights\n(batch, heads, seq, seq)"]
    ATTENDED["Attended = Weights · V\n(batch, heads, seq, head_dim)"]
    MERGE["Merge Heads → Concat\n(batch, seq, embed_dim)"]
    OUT["Output Projection · W_O\n(batch, seq, embed_dim)"]

    X --> Q & K & V
    Q & K --> SCORES
    MASK --> SCORES
    SCORES --> SOFTMAX
    SOFTMAX & V --> ATTENDED
    ATTENDED --> MERGE --> OUT
```

### Transformer Encoder Block

Each encoder block applies self-attention followed by a position-wise feed-forward network (FFN). Both sub-layers use **Pre-LN style** residual connections - a residual skip connection wraps each sub-layer, and layer normalization is applied after the addition. Dropout is applied to the output of each sub-layer before the residual addition, acting as a regularizer.

```mermaid
flowchart LR
    IN["Input x\n(batch, seq, D)"]
    ATTN["Multi-Head\nSelf-Attention"]
    DROP1["Dropout"]
    ADD1["Add + LayerNorm"]
    FFN["Position-wise FFN\nLinear → ReLU → Dropout → Linear"]
    DROP2["Dropout"]
    ADD2["Add + LayerNorm"]
    OUT["Output\n(batch, seq, D)"]

    IN --> ATTN --> DROP1 --> ADD1 --> FFN --> DROP2 --> ADD2 --> OUT
    IN -->|"residual"| ADD1
    ADD1 -->|"residual"| ADD2
```

### Full Model Forward Pass

The complete `TransformerSequenceClassifier` stacks two encoder blocks. Token IDs and their positions are each embedded and summed to produce the initial representation. After the encoder stack, mean pooling collapses the sequence dimension, and a small two-layer MLP head maps the pooled vector to class logits.

```mermaid
flowchart TD
    TOKENS["Input Tokens\n(batch, seq_len) - integer IDs"]
    TOK_EMB["Token Embedding\n nn.Embedding(vocab_size, embed_dim)"]
    POS_EMB["Positional Embedding\n nn.Embedding(max_seq_len, embed_dim)"]
    SUM["Sum Embeddings\n(batch, seq_len, embed_dim)"]
    B1["Encoder Block 1\nSelf-Attn + FFN + ResNorm"]
    B2["Encoder Block 2\nSelf-Attn + FFN + ResNorm"]
    POOL["Mean Pool over Sequence\n(batch, embed_dim)"]
    HEAD["Classifier Head\nLinear → ReLU → Dropout → Linear"]
    LOGITS["Logits\n(batch, num_classes=2)"]
    MAPS["Attention Maps\nReturned for visualization"]

    TOKENS --> TOK_EMB & POS_EMB
    TOK_EMB & POS_EMB --> SUM
    SUM --> B1 --> B2
    B1 & B2 -->|"attn_weights"| MAPS
    B2 --> POOL --> HEAD --> LOGITS
```

### Training Pipeline

Both the Transformer and the Baseline are trained using the same data splits, optimizer family, and loss function so the comparison is apples-to-apples. Each epoch runs a full pass over the training set with gradient updates, followed by a validation pass in `torch.no_grad()` context. Metrics are collected per epoch and persisted at the end.

```mermaid
sequenceDiagram
    participant CLI as CLI (main.py)
    participant D as data.py
    participant T as train.py
    participant M as model/
    participant A as artifacts/

    CLI->>D: build_dataloaders(config)
    D-->>CLI: train_loader, val_loader, test_loader
    CLI->>T: train_models(config)
    loop Each Epoch
        T->>M: forward pass (train=True)
        M-->>T: logits, attn_maps
        T->>T: loss.backward() + optimizer.step()
        T->>M: forward pass (train=False, val set)
        M-->>T: val logits
    end
    T->>A: save transformer.pt, baseline.pt
    T->>A: save metrics.json
```

### Comparison Architecture

The baseline model uses the same token and positional embedding as the Transformer, but replaces the encoder stack with a direct MLP applied to the mean-pooled embedding. This is equivalent to a bag-of-embeddings model. It cannot learn positional structure and cannot directly compare distant tokens, which is why it systematically underperforms on the first-equals-last task.

```mermaid
flowchart LR
    subgraph TRANSFORMER["TransformerSequenceClassifier"]
        direction TB
        TE["Embed (tok + pos)"] --> TB1["Block 1"] --> TB2["Block 2"] --> TP["Mean Pool"] --> TH["MLP Head"] --> TL["Logits"]
    end
    subgraph BASELINE["BaselineSequenceClassifier"]
        direction TB
        BE["Embed (tok + pos)"] --> BP["Mean Pool"] --> BH["MLP Head"] --> BL["Logits"]
    end
```

> [!TIP]
> Open `notebooks/attention_exploration.ipynb` after training to interactively inspect which token pairs receive high attention weights across both layers and all four heads. You will see that the model learns to allocate attention to position 0 and position $L-1$ to solve the task.

---

## Tech Stack

| # | <sub>Library</sub> | <sub>Version</sub> | <sub>Role</sub> | <sub>Why it was chosen</sub> |
|---|---|---|---|---|
| 1 | <sub>PyTorch</sub> | <sub>≥ 2.2</sub> | <sub>Core deep learning framework</sub> | <sub>Imperative, debuggable, industry-standard for research</sub> |
| 2 | <sub>NumPy</sub> | <sub>≥ 1.26</sub> | <sub>Array utilities</sub> | <sub>Used for dataset generation and metric arrays</sub> |
| 3 | <sub>Matplotlib</sub> | <sub>≥ 3.8</sub> | <sub>Plotting</sub> | <sub>Produces attention heatmaps and training curves</sub> |
| 4 | <sub>Seaborn</sub> | <sub>≥ 0.13</sub> | <sub>Statistical plot styling</sub> | <sub>Renders attention weight heatmaps with clean color scales</sub> |
| 5 | <sub>tqdm</sub> | <sub>≥ 4.66</sub> | <sub>Progress bars</sub> | <sub>Epoch and batch-level progress display in the terminal</sub> |

> [!NOTE]
> All five dependencies are CPU-compatible. No GPU is required to run this project. PyTorch will automatically detect and use CUDA or MPS if available.

---

## Project Structure

```text
transformer-demo/
├── .github/
│   ├── workflows/ci.yml          - GitHub Actions CI
│   ├── ISSUE_TEMPLATE/           - Bug and feature issue forms
│   ├── pull_request_template.md  - PR checklist
│   ├── CONTRIBUTING.md
│   ├── SECURITY.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── artifacts/                    - Saved checkpoints and metrics (git-ignored)
│   ├── transformer.pt
│   ├── baseline.pt
│   └── metrics.json
├── notebooks/
│   └── attention_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py                   - Dataset generation and DataLoader construction
│   ├── evaluate.py               - Evaluation logic and metrics
│   ├── main.py                   - CLI entry point (train / evaluate / visualize)
│   ├── train.py                  - Training loop for both models
│   ├── utils.py                  - Shared helpers (device, seed, I/O)
│   ├── visualize.py              - Attention heatmaps and comparison plots
│   └── model/
│       ├── __init__.py
│       ├── attention.py          - ScaledDotProductAttention, MultiHeadSelfAttention
│       ├── transformer_block.py  - PositionwiseFeedForward, TransformerEncoderBlock
│       └── transformer_model.py  - TransformerSequenceClassifier, BaselineSequenceClassifier
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

Python 3.11 or higher is recommended. A virtual environment is strongly recommended to avoid version conflicts with system packages.

```bash
cd /home/kevin/Projects/transformer-demo
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> If you have a CUDA-capable GPU, replace the plain `torch` install with a CUDA-enabled wheel from [pytorch.org](https://pytorch.org/get-started/locally/). The code will automatically use the GPU without any further changes.

> [!WARNING]
> Do not run the training or evaluation scripts outside of the virtual environment. The project's exact dependency versions are pinned in `requirements.txt` to ensure reproducibility.

---

## Usage

All commands are routed through `src/main.py` using Python's `-m` flag, which ensures the `src` package is correctly on the path.

### Train

Train both the Transformer and the Baseline on freshly generated synthetic data. Progress bars show per-batch loss and accuracy. At the end of each epoch, validation metrics are printed. Both models and all metrics are saved to `artifacts/`.

```bash
python -m src.main train --epochs 15 --batch-size 64 --seq-len 16
```

### Evaluate

Load the saved checkpoints and run evaluation on the held-out test set. Prints accuracy and cross-entropy loss for both models side by side.

```bash
python -m src.main evaluate
```

### Visualize

Generate attention heatmap images for every layer and head of the Transformer on a selected test sample, plus a side-by-side training curve comparison chart.

```bash
python -m src.main visualize --sample-index 0
```

> [!NOTE]
> You must run `train` before `evaluate` or `visualize`. The evaluate and visualize commands load weights from `artifacts/transformer.pt` and `artifacts/baseline.pt`, which are produced only after training completes.

---

## Configuration Reference

### Train Command Options

| # | <sub>Flag</sub> | <sub>Type</sub> | <sub>Default</sub> | <sub>Description</sub> |
|---|---|---|---|---|
| 1 | <sub>`--epochs`</sub> | <sub>int</sub> | <sub>15</sub> | <sub>Number of full passes over the training set</sub> |
| 2 | <sub>`--batch-size`</sub> | <sub>int</sub> | <sub>64</sub> | <sub>Samples per gradient update step</sub> |
| 3 | <sub>`--seq-len`</sub> | <sub>int</sub> | <sub>16</sub> | <sub>Length of each token sequence</sub> |
| 4 | <sub>`--vocab-size`</sub> | <sub>int</sub> | <sub>20</sub> | <sub>Number of distinct token IDs (0 to vocab-1)</sub> |
| 5 | <sub>`--embed-dim`</sub> | <sub>int</sub> | <sub>64</sub> | <sub>Dimensionality of token and positional embeddings</sub> |
| 6 | <sub>`--num-heads`</sub> | <sub>int</sub> | <sub>4</sub> | <sub>Number of parallel attention heads</sub> |
| 7 | <sub>`--ff-hidden-dim`</sub> | <sub>int</sub> | <sub>128</sub> | <sub>Hidden layer size in the position-wise FFN</sub> |
| 8 | <sub>`--num-layers`</sub> | <sub>int</sub> | <sub>2</sub> | <sub>Number of stacked Transformer encoder blocks</sub> |
| 9 | <sub>`--dropout`</sub> | <sub>float</sub> | <sub>0.1</sub> | <sub>Dropout probability applied after attention and FFN</sub> |
| 10 | <sub>`--lr`</sub> | <sub>float</sub> | <sub>1e-3</sub> | <sub>Adam optimizer learning rate</sub> |
| 11 | <sub>`--weight-decay`</sub> | <sub>float</sub> | <sub>1e-4</sub> | <sub>L2 regularization coefficient in Adam</sub> |
| 12 | <sub>`--train-size`</sub> | <sub>int</sub> | <sub>4000</sub> | <sub>Number of training samples to generate</sub> |
| 13 | <sub>`--val-size`</sub> | <sub>int</sub> | <sub>800</sub> | <sub>Number of validation samples</sub> |
| 14 | <sub>`--test-size`</sub> | <sub>int</sub> | <sub>800</sub> | <sub>Number of test samples (used in evaluate)</sub> |
| 15 | <sub>`--seed`</sub> | <sub>int</sub> | <sub>42</sub> | <sub>Random seed for dataset and model initialization</sub> |
| 16 | <sub>`--artifacts-dir`</sub> | <sub>str</sub> | <sub>artifacts</sub> | <sub>Directory for saved checkpoints and metrics</sub> |

### Evaluate Command Options

| # | <sub>Flag</sub> | <sub>Type</sub> | <sub>Default</sub> | <sub>Description</sub> |
|---|---|---|---|---|
| 1 | <sub>`--batch-size`</sub> | <sub>int</sub> | <sub>64</sub> | <sub>Samples per evaluation batch</sub> |
| 2 | <sub>`--seq-len`</sub> | <sub>int</sub> | <sub>16</sub> | <sub>Sequence length - must match training</sub> |
| 3 | <sub>`--vocab-size`</sub> | <sub>int</sub> | <sub>20</sub> | <sub>Vocabulary size - must match training</sub> |
| 4 | <sub>`--embed-dim`</sub> | <sub>int</sub> | <sub>64</sub> | <sub>Embedding dim - must match training</sub> |
| 5 | <sub>`--num-heads`</sub> | <sub>int</sub> | <sub>4</sub> | <sub>Attention heads - must match training</sub> |
| 6 | <sub>`--num-layers`</sub> | <sub>int</sub> | <sub>2</sub> | <sub>Encoder depth - must match training</sub> |
| 7 | <sub>`--test-size`</sub> | <sub>int</sub> | <sub>800</sub> | <sub>Number of test samples to regenerate</sub> |
| 8 | <sub>`--seed`</sub> | <sub>int</sub> | <sub>42</sub> | <sub>Seed - must match training to reproduce test set</sub> |

> [!IMPORTANT]
> The `--seed`, `--seq-len`, `--vocab-size`, `--embed-dim`, `--num-heads`, `--ff-hidden-dim`, and `--num-layers` flags passed to `evaluate` and `visualize` must exactly match the values used during `train`. Mismatching these will cause a shape error when loading the checkpoint.

### Visualize Command Options

| # | <sub>Flag</sub> | <sub>Type</sub> | <sub>Default</sub> | <sub>Description</sub> |
|---|---|---|---|---|
| 1 | <sub>`--sample-index`</sub> | <sub>int</sub> | <sub>0</sub> | <sub>Index in the test set to visualize attention for</sub> |
| 2 | <sub>`--seq-len`</sub> | <sub>int</sub> | <sub>16</sub> | <sub>Sequence length - must match training</sub> |
| 3 | <sub>`--vocab-size`</sub> | <sub>int</sub> | <sub>20</sub> | <sub>Vocabulary size - must match training</sub> |
| 4 | <sub>`--embed-dim`</sub> | <sub>int</sub> | <sub>64</sub> | <sub>Embedding dim - must match training</sub> |
| 5 | <sub>`--num-heads`</sub> | <sub>int</sub> | <sub>4</sub> | <sub>Number of heads - controls how many heatmaps are produced</sub> |
| 6 | <sub>`--num-layers`</sub> | <sub>int</sub> | <sub>2</sub> | <sub>Encoder depth - controls how many layer heatmaps are produced</sub> |

---

## Outputs and Artifacts

After running `train` and `visualize`, the `artifacts/` directory will contain the following files. Checkpoint files are excluded from Git via `.gitignore` to keep the repository lightweight.

| # | <sub>File</sub> | <sub>Produced by</sub> | <sub>Contents</sub> |
|---|---|---|---|
| 1 | <sub>`transformer.pt`</sub> | <sub>train</sub> | <sub>Full `state_dict` of `TransformerSequenceClassifier`</sub> |
| 2 | <sub>`baseline.pt`</sub> | <sub>train</sub> | <sub>Full `state_dict` of `BaselineSequenceClassifier`</sub> |
| 3 | <sub>`metrics.json`</sub> | <sub>train</sub> | <sub>Per-epoch train/val loss and accuracy for both models</sub> |
| 4 | <sub>`attention_heatmap_layer0.png`</sub> | <sub>visualize</sub> | <sub>Grid of attention weight heatmaps for encoder block 1</sub> |
| 5 | <sub>`attention_heatmap_layer1.png`</sub> | <sub>visualize</sub> | <sub>Grid of attention weight heatmaps for encoder block 2</sub> |
| 6 | <sub>`training_comparison.png`</sub> | <sub>visualize</sub> | <sub>Side-by-side training and validation curves for both models</sub> |

---

## Model Comparison

The table below summarizes the design differences between the two models. The expected accuracy gap reflects typical results on the default configuration after 15 epochs.

| # | <sub>Aspect</sub> | <sub>TransformerSequenceClassifier</sub> | <sub>BaselineSequenceClassifier</sub> |
|---|---|---|---|
| 1 | <sub>Embedding</sub> | <sub>Token + Positional (summed)</sub> | <sub>Token + Positional (summed)</sub> |
| 2 | <sub>Context mechanism</sub> | <sub>Multi-head self-attention</sub> | <sub>None</sub> |
| 3 | <sub>Sequence processing</sub> | <sub>2 x Encoder block (Attn + FFN + ResNorm)</sub> | <sub>Direct mean pool</sub> |
| 4 | <sub>Classifier head</sub> | <sub>Linear → ReLU → Dropout → Linear</sub> | <sub>Linear → ReLU → Dropout → Linear</sub> |
| 5 | <sub>Trainable parameters</sub> | <sub>~50K (default config)</sub> | <sub>~10K (default config)</sub> |
| 6 | <sub>Expected test accuracy</sub> | <sub>~95-99%</sub> | <sub>~55-65% (near chance)</sub> |
| 7 | <sub>Attention maps</sub> | <sub>Yes - returned and visualized</sub> | <sub>No</sub> |
| 8 | <sub>Suitable for long-range tasks</sub> | <sub>Yes</sub> | <sub>No</sub> |

> [!NOTE]
> The baseline model hovers near chance on this task because mean pooling destroys all positional structure. After averaging embeddings, no information remains about which tokens were at the first or last position. This is exactly the failure mode that positional attention was designed to address.

---

## Notebook Exploration

The Jupyter notebook `notebooks/attention_exploration.ipynb` provides an interactive environment for loading a trained model, running inference on custom sequences, and rendering rich attention visualizations inline. It is the best way to build intuition for how attention weights evolve across layers and heads and how they change between correctly and incorrectly classified examples.

> [!TIP]
> To launch the notebook, install `jupyter` in your virtual environment (`pip install jupyter`) and run `jupyter notebook notebooks/attention_exploration.ipynb`. The notebook assumes training has already been completed and `artifacts/transformer.pt` exists.

---

## API Reference

<details>
<summary><strong>src.model.attention - ScaledDotProductAttention</strong></summary>

`ScaledDotProductAttention(dropout=0.0)`

Computes scaled dot-product attention as defined in Vaswani et al. (2017). Expects pre-split head tensors.

**forward(q, k, v, mask=None)**

| # | <sub>Argument</sub> | <sub>Shape</sub> | <sub>Description</sub> |
|---|---|---|---|
| 1 | <sub>`q`</sub> | <sub>(batch, heads, tgt_len, head_dim)</sub> | <sub>Query tensor from Q projection</sub> |
| 2 | <sub>`k`</sub> | <sub>(batch, heads, src_len, head_dim)</sub> | <sub>Key tensor from K projection</sub> |
| 3 | <sub>`v`</sub> | <sub>(batch, heads, src_len, head_dim)</sub> | <sub>Value tensor from V projection</sub> |
| 4 | <sub>`mask`</sub> | <sub>broadcastable to (batch, heads, tgt, src)</sub> | <sub>`bool` (True=attend) or float additive mask</sub> |

Returns `(attended, attn_weights)` both of shape `(batch, heads, tgt_len, head_dim/src_len)`.

</details>

<details>
<summary><strong>src.model.attention - MultiHeadSelfAttention</strong></summary>

`MultiHeadSelfAttention(embed_dim, num_heads, dropout=0.0)`

Projects the input into Q, K, V spaces, splits into heads, runs scaled dot-product attention in each head, and merges the results back to `embed_dim`. Requires `embed_dim % num_heads == 0`.

**forward(x, mask=None)**

| # | <sub>Argument</sub> | <sub>Shape</sub> | <sub>Description</sub> |
|---|---|---|---|
| 1 | <sub>`x`</sub> | <sub>(batch, seq, embed_dim)</sub> | <sub>Input sequence representation</sub> |
| 2 | <sub>`mask`</sub> | <sub>(L,S), (B,L,S), or (B,H,L,S)</sub> | <sub>Optional attention mask</sub> |

Returns `(output, attn_weights)` where `output` is `(batch, seq, embed_dim)` and `attn_weights` is `(batch, heads, seq, seq)`.

</details>

<details>
<summary><strong>src.model.transformer_block - TransformerEncoderBlock</strong></summary>

`TransformerEncoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout=0.1)`

One complete Transformer encoder layer consisting of multi-head self-attention followed by a position-wise FFN, each wrapped in a residual + LayerNorm sub-layer. This is the building block that is stacked `num_layers` times in `TransformerSequenceClassifier`.

**forward(x, mask=None)**

| # | <sub>Argument</sub> | <sub>Shape</sub> | <sub>Description</sub> |
|---|---|---|---|
| 1 | <sub>`x`</sub> | <sub>(batch, seq, embed_dim)</sub> | <sub>Sequence from previous layer or embedding</sub> |
| 2 | <sub>`mask`</sub> | <sub>Optional</sub> | <sub>Passed through to MultiHeadSelfAttention</sub> |

Returns `(x, attn_weights)`.

</details>

<details>
<summary><strong>src.model.transformer_model - TransformerSequenceClassifier</strong></summary>

`TransformerSequenceClassifier(vocab_size, max_seq_len, embed_dim=64, num_heads=4, ff_hidden_dim=128, num_layers=2, num_classes=2, dropout=0.1)`

The full Transformer encoder classifier. Sums token and positional embeddings, applies `num_layers` encoder blocks, mean-pools the output sequence, and projects to `num_classes` logits. Stores all per-layer attention maps on `self.attention_maps` after each forward pass.

**forward(tokens, mask=None)**

| # | <sub>Argument</sub> | <sub>Shape</sub> | <sub>Description</sub> |
|---|---|---|---|
| 1 | <sub>`tokens`</sub> | <sub>(batch, seq_len)</sub> | <sub>Integer token IDs in range [0, vocab_size)</sub> |
| 2 | <sub>`mask`</sub> | <sub>Optional</sub> | <sub>Optional attention mask</sub> |

Returns `(logits, attention_maps)` where `logits` is `(batch, num_classes)` and `attention_maps` is a list of `num_layers` tensors each shaped `(batch, num_heads, seq, seq)`.

</details>

<details>
<summary><strong>src.model.transformer_model - BaselineSequenceClassifier</strong></summary>

`BaselineSequenceClassifier(vocab_size, max_seq_len, embed_dim=64, num_classes=2, dropout=0.1)`

A simple bag-of-embeddings classifier that uses the same token and positional embedding as the Transformer but skips the attention stack entirely. It mean-pools the embeddings and applies an MLP head. Exists purely as a controlled comparison.

**forward(tokens)**

| # | <sub>Argument</sub> | <sub>Shape</sub> | <sub>Description</sub> |
|---|---|---|---|
| 1 | <sub>`tokens`</sub> | <sub>(batch, seq_len)</sub> | <sub>Integer token IDs in range [0, vocab_size)</sub> |

Returns `logits` of shape `(batch, num_classes)`.

</details>

<details>
<summary><strong>src.data - DatasetConfig and build_dataloaders</strong></summary>

`DatasetConfig` is a dataclass that holds all dataset generation parameters (vocab_size, seq_len, train/val/test sizes, batch_size, seed). `build_dataloaders(config)` generates fully synthetic data and returns `(train_loader, val_loader, test_loader)`.

The dataset generation logic is: sample random integer sequences uniformly from `[0, vocab_size)`, then assign label `1` if `tokens[0] == tokens[-1]` and `0` otherwise. This produces a roughly 1/vocab_size positive rate (e.g., 5% positives for vocab_size=20), so class weights or oversampling are used to balance the training set.

</details>

---

> [!CAUTION]
> The `artifacts/` directory is excluded from Git by `.gitignore`. If you clone a fresh copy of this repository, you must run `python -m src.main train` before you can evaluate or visualize anything. There are no pre-trained weights committed to the repository.

---

<div align="center">

Built to make Transformer internals inspectable, not magical.

</div>
