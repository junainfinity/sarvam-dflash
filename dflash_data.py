"""
DFlash training data generation and dataset utilities.

Phase 1 (offline): Run frozen Sarvam-30B on a text corpus to pre-extract:
  - KV injection features (5-layer hidden state concatenation) [B, S, 20480]
  - Top-k teacher logits for white-box distillation [B, S, k] + indices

Phase 2 (training): Load pre-extracted data from disk.

This offline approach is 11-25% better than online distillation
(per "Training Domain Draft Models for Speculative Decoding", arXiv 2503.07807).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from tqdm import tqdm

# sarvam-30b imports are handled via modeling_sarvam_moe_dflash


# ---------------------------------------------------------------------------
# Offline Data Generation
# ---------------------------------------------------------------------------

def generate_training_data(
    target_model_path: str,
    dataset_name: str,
    output_dir: str,
    max_samples: int = 50000,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    teacher_top_k: int = 64,
    dataset_split: str = "train",
    dataset_text_field: str = "text",
):
    """
    Generate pre-extracted features from the frozen Sarvam-30B target model.

    Saves per-shard files to output_dir:
      shard_{i}.safetensors containing:
        - input_ids: [B, S] int32
        - injection_features: [B, S, 20480] bfloat16
        - teacher_top_logits: [B, S, k] float32
        - teacher_top_indices: [B, S, k] int32
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from modeling_sarvam_moe_dflash import SarvamMoEForCausalLMWithKVInjection, SarvamMoEConfig

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {target_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load target model
    print(f"Loading Sarvam-30B from {target_model_path}...")
    config = SarvamMoEConfig.from_pretrained(target_model_path)
    model = SarvamMoEForCausalLMWithKVInjection.from_pretrained(
        target_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Target model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")

    # Load and tokenize dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)

    # Resume support: count existing shards to skip ahead
    existing_shards = sorted(Path(output_dir).glob("shard_*.safetensors"))
    resume_shard_idx = len(existing_shards)
    resume_sample_count = resume_shard_idx * batch_size  # approximate

    if resume_shard_idx > 0:
        print(f"Resuming: found {resume_shard_idx} existing shards ({resume_sample_count} samples). Skipping ahead...")

    shard_idx = 0
    sample_count = 0
    batch_input_ids = []

    for sample in tqdm(dataset, total=max_samples, desc="Processing samples"):
        if sample_count >= max_samples:
            break

        text = sample.get(dataset_text_field, "")
        if not text or len(text) < 100:
            continue

        tokens = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)  # [S]

        num_real_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
        if num_real_tokens < 32:
            continue

        batch_input_ids.append(input_ids)
        sample_count += 1

        if len(batch_input_ids) == batch_size:
            # Skip shards that already exist (resume support)
            if shard_idx < resume_shard_idx:
                shard_idx += 1
                batch_input_ids = []
                continue

            _process_and_save_batch(
                model, batch_input_ids, output_dir, shard_idx,
                teacher_top_k, max_seq_len,
            )
            shard_idx += 1
            batch_input_ids = []

    # Process remaining
    if batch_input_ids and shard_idx >= resume_shard_idx:
        _process_and_save_batch(
            model, batch_input_ids, output_dir, shard_idx,
            teacher_top_k, max_seq_len,
        )
        shard_idx += 1

    # Save/update metadata
    metadata = {
        "num_shards": shard_idx,
        "max_seq_len": max_seq_len,
        "teacher_top_k": teacher_top_k,
        "total_samples": sample_count,
        "injection_dim": 20480,
        "vocab_size": 262144,
    }
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))
    print(f"Data generation complete: {sample_count} samples in {shard_idx} shards → {output_dir}")


@torch.no_grad()
def _process_and_save_batch(
    model, batch_input_ids, output_dir, shard_idx, teacher_top_k, max_seq_len,
):
    """Process a batch through the target model and save features."""
    # With device_map="auto", get the device of the first parameter
    first_device = next(model.parameters()).device
    input_ids = torch.stack(batch_input_ids).to(first_device)  # [B, S]

    # Single forward pass for both injection features and teacher logits
    injection_features, top_logits, top_indices = model.get_injection_and_logits(
        input_ids=input_ids,
        top_k=teacher_top_k,
    )

    # Save as safetensors (move to CPU, convert dtypes for storage)
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.safetensors")
    save_file(
        {
            "input_ids": input_ids.cpu().to(torch.int32),
            "injection_features": injection_features.cpu().to(torch.bfloat16),
            "teacher_top_logits": top_logits.cpu().to(torch.float32),
            "teacher_top_indices": top_indices.cpu().to(torch.int32),
        },
        shard_path,
    )


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class DFlashDataset(Dataset):
    """
    Loads pre-extracted DFlash training data from safetensors shards.

    Each item returns:
        input_ids: [seq_len] int64
        injection_features: [seq_len, 20480] bfloat16
        teacher_top_logits: [seq_len, top_k] float32
        teacher_top_indices: [seq_len, top_k] int64
    """

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len

        # Load metadata
        metadata = torch.load(self.data_dir / "metadata.pt", weights_only=True)
        self.num_shards = metadata["num_shards"]
        self.teacher_top_k = metadata["teacher_top_k"]

        # Build index: (shard_idx, sample_idx_within_shard)
        # Each shard has batch_size=1 (from data gen with --batch_size 1),
        # so we just list existing shards without loading them.
        self.index = []
        for shard_idx in range(self.num_shards):
            shard_path = self.data_dir / f"shard_{shard_idx:06d}.safetensors"
            if shard_path.exists():
                self.index.append((shard_idx, 0))

        # Cache for current shard (avoid re-loading for sequential access)
        self._cached_shard_idx = -1
        self._cached_data = None

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_idx: int):
        if shard_idx != self._cached_shard_idx:
            shard_path = self.data_dir / f"shard_{shard_idx:06d}.safetensors"
            self._cached_data = load_file(str(shard_path))
            self._cached_shard_idx = shard_idx

    def __getitem__(self, idx):
        shard_idx, sample_idx = self.index[idx]
        self._load_shard(shard_idx)

        return {
            "input_ids": self._cached_data["input_ids"][sample_idx].to(torch.long),
            "injection_features": self._cached_data["injection_features"][sample_idx],
            "teacher_top_logits": self._cached_data["teacher_top_logits"][sample_idx],
            "teacher_top_indices": self._cached_data["teacher_top_indices"][sample_idx].to(torch.long),
        }


# ---------------------------------------------------------------------------
# CLI Entry Point for Data Generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DFlash training data from Sarvam-30B")
    parser.add_argument("--target_model_path", type=str, required=True, help="Path to Sarvam-30B model directory")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu-score-2",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted features")
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--teacher_top_k", type=int, default=64)

    args = parser.parse_args()
    generate_training_data(
        target_model_path=args.target_model_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        teacher_top_k=args.teacher_top_k,
        dataset_split=args.dataset_split,
        dataset_text_field=args.dataset_text_field,
    )
