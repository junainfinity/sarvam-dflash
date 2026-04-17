"""
Load the converted DFlash MLX draft model and run inference.

Usage:
    from load import load_model
    model = load_model()

    # Or from command line for a quick test:
    python3 load.py
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model import DFlashConfig, DFlashDraftModel, make_block_causal_mask


def load_model(model_dir: str = ".") -> DFlashDraftModel:
    """Load DFlash MLX model from converted weights."""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)
    config = DFlashConfig(**config_dict)

    # Create model
    model = DFlashDraftModel(config)

    # Load weights
    weights_path = model_dir / "weights.npz"
    print(f"Loading weights from {weights_path}...")
    data = np.load(str(weights_path), allow_pickle=False)

    weight_dict = {}
    for key in data.files:
        weight_dict[key] = mx.array(data[key])

    # Apply weights to model
    model.load_weights(list(weight_dict.items()))

    mx.eval(model.parameters())
    from mlx.utils import tree_flatten
    print(f"Model loaded: {sum(v.size for _, v in tree_flatten(model.parameters())) / 1e6:.1f}M parameters")
    return model


def test_forward(model_dir: str = "."):
    """Quick smoke test: load model, run forward pass with random data."""
    model = load_model(model_dir)

    # Random inputs
    B, S = 1, 32
    input_ids = mx.array(np.random.randint(0, 1000, (B, S)))
    injection_feats = mx.random.normal((B, S, 20480))
    mask = make_block_causal_mask(S, 16)

    print(f"Running forward pass (B={B}, S={S})...")
    logits = model(input_ids, injection_feats, mask)
    mx.eval(logits)
    print(f"Output: {logits.shape}, dtype={logits.dtype}")
    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print("MLX inference OK!")


if __name__ == "__main__":
    test_forward()
