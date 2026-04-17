"""
Convert PyTorch DFlash draft model checkpoint + Sarvam-30B shared weights to MLX.

Outputs:
  dflash_mlx/
    model.py          — MLX model definition
    weights.npz       — all weights (draft + shared embed/head)
    config.json       — model config
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert(
    checkpoint_path: str = "../checkpoints/dflash_draft_best.pt",
    sarvam_path: str = "../sarvam-30b",
    output_dir: str = ".",
):
    import torch
    from safetensors.torch import load_file as st_load

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load draft model weights from PyTorch checkpoint
    # -----------------------------------------------------------------------
    print(f"Loading draft checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    # Support both full-state and model-only checkpoints
    draft_state = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    config = ckpt["config"]
    print(f"  {len(draft_state)} draft tensors, loss={ckpt.get('loss', 'N/A')}")

    # -----------------------------------------------------------------------
    # 2. Load shared embedding + lm_head from Sarvam-30B safetensors
    # -----------------------------------------------------------------------
    print(f"Loading shared weights from: {sarvam_path}")
    sarvam_path = Path(sarvam_path)

    embed_shard = st_load(str(sarvam_path / "model-00001-of-00026.safetensors"))
    embed_weight = embed_shard["model.word_embeddings.weight"].to(torch.float16).numpy()
    del embed_shard
    print(f"  word_embeddings: {embed_weight.shape}")

    head_shard = st_load(str(sarvam_path / "model-00026-of-00026.safetensors"))
    head_weight = head_shard["lm_head.weight"].to(torch.float16).numpy()
    del head_shard
    print(f"  lm_head: {head_weight.shape}")

    # -----------------------------------------------------------------------
    # 3. Map PyTorch key names → MLX key names
    # -----------------------------------------------------------------------
    mlx_weights = {}

    # Shared weights
    mlx_weights["word_embeddings.weight"] = embed_weight
    mlx_weights["lm_head.weight"] = head_weight

    # Draft trainable weights
    for pt_key, tensor in draft_state.items():
        arr = tensor.to(torch.float16).numpy()

        # Map PyTorch names to MLX names
        # PyTorch: kv_fusion.kv_proj.weight → MLX: kv_fusion.kv_proj.weight
        # PyTorch: layers.0.input_layernorm.weight → MLX: layers.0.input_layernorm.weight
        # PyTorch: layers.0.self_attn.query_key_value.weight → MLX: layers.0.self_attn.query_key_value.weight
        # etc. — names are identical since we used the same structure.

        # nn.Linear in MLX stores weight as [out, in] same as PyTorch
        mlx_weights[pt_key] = arr

    # RMSNorm weights (from PyTorch SarvamMoERMSNorm) — already named correctly
    # rotary_emb inv_freq is computed on-the-fly in MLX, not stored

    print(f"  Total: {len(mlx_weights)} tensors")

    # -----------------------------------------------------------------------
    # 4. Save as .npz
    # -----------------------------------------------------------------------
    weights_path = output_dir / "weights.npz"
    print(f"Saving weights to {weights_path}...")
    np.savez(str(weights_path), **mlx_weights)
    size_mb = weights_path.stat().st_size / 1e6
    print(f"  Saved: {size_mb:.1f} MB")

    # -----------------------------------------------------------------------
    # 5. Save config
    # -----------------------------------------------------------------------
    config_dict = {
        "hidden_size": config.hidden_size,
        "num_draft_layers": config.num_draft_layers,
        "num_self_attn_heads": config.num_self_attn_heads,
        "num_self_attn_kv_heads": config.num_self_attn_kv_heads,
        "num_cross_attn_heads": config.num_cross_attn_heads,
        "num_cross_attn_kv_heads": config.num_cross_attn_kv_heads,
        "head_dim": config.head_dim,
        "ffn_intermediate": config.ffn_intermediate,
        "block_size": config.block_size,
        "injection_dim": config.injection_dim,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "vocab_size": config.vocab_size,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Config saved to {config_path}")

    print("\nConversion complete!")
    print(f"  MLX model dir: {output_dir}")
    print(f"  To load: python3 -c \"from load import load_model; model = load_model()\"")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/dflash_draft_best.pt")
    parser.add_argument("--sarvam", default="../sarvam-30b")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    convert(args.checkpoint, args.sarvam, args.output)
