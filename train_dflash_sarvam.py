"""
DFlash Draft Model Training Script for Sarvam-30B.

Trains a ~275M parameter block diffusion draft model using:
- Offline pre-extracted KV injection features and teacher logits
- Block-diagonal causal self-attention (block_size=16)
- Combined loss: position-weighted CE + KL distillation (white-box)
- AdamW optimizer with cosine LR schedule
- **Full pause/resume**: saves model, optimizer, scheduler, RNG states,
  epoch, batch index, and best_loss every N steps. On restart, resumes
  from the exact step with minimal lost progress.

Usage:
    torchrun --nproc_per_node=1 train_dflash_sarvam.py \
        --target_model_path ./sarvam-30b \
        --data_dir ./dflash_training_data \
        --output_dir ./checkpoints

    # To resume after interruption, just run the same command again.
    # It auto-detects the latest checkpoint and resumes.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

from dflash_draft import DFlashConfig, DFlashDraftModel, make_block_causal_mask, make_sequence_position_weights
from dflash_data import DFlashDataset
from modeling_sarvam_moe_dflash import SarvamMoEForCausalLMWithKVInjection, SarvamMoEConfig


# ---------------------------------------------------------------------------
# Graceful shutdown handler
# ---------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False


def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        print("\nForce quit.")
        sys.exit(1)
    print("\nShutdown requested — will save checkpoint after current step...")
    _SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def chunked_cross_entropy(
    logits: torch.Tensor,  # [B, S, V]
    targets: torch.Tensor,  # [B, S]
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Memory-efficient cross-entropy for large vocab (262K).
    Returns per-token loss [B, S].
    """
    B, S, V = logits.shape
    logits_flat = logits.reshape(-1, V).float()
    targets_flat = targets.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=ignore_index)
    return loss.reshape(B, S)


def sparse_kl_divergence(
    draft_logits: torch.Tensor,      # [B, S, V]
    teacher_vals: torch.Tensor,      # [B, S, k]
    teacher_idxs: torch.Tensor,      # [B, S, k]
) -> torch.Tensor:
    """
    KL divergence on top-k teacher tokens only (memory efficient).
    Returns per-token KL [B, S].
    """
    draft_at_teacher = torch.gather(draft_logits, -1, teacher_idxs)  # [B, S, k]
    teacher_probs = F.softmax(teacher_vals.float(), dim=-1)
    draft_log_probs = F.log_softmax(draft_at_teacher.float(), dim=-1)
    kl = (teacher_probs * (teacher_probs.log().clamp(min=-100) - draft_log_probs)).sum(dim=-1)
    return kl


# ---------------------------------------------------------------------------
# Checkpoint save/load (full training state)
# ---------------------------------------------------------------------------

RESUME_CHECKPOINT_NAME = "resume_checkpoint.pt"


def save_full_checkpoint(
    draft_model, draft_config, optimizer, scheduler,
    epoch, batch_idx, global_step, best_loss,
    output_dir, tag=None,
):
    """
    Save complete training state for exact resumption.

    Saves:
      - Draft model trainable weights
      - Optimizer state dict (momentum buffers, etc.)
      - Scheduler state dict
      - CUDA and Python RNG states
      - Epoch, batch_idx, global_step, best_loss
    """
    # Collect trainable state dict
    model_state = {}
    for name, param in draft_model.named_parameters():
        if param.requires_grad:
            model_state[name] = param.data.cpu()

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": draft_config,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "best_loss": best_loss,
        "python_rng_state": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        checkpoint["mps_available"] = True  # MPS doesn't expose RNG state directly

    # Always save as resume checkpoint (overwritten each time for latest state)
    resume_path = os.path.join(output_dir, RESUME_CHECKPOINT_NAME)
    # Write to temp file first, then atomic rename to prevent corruption on crash
    tmp_path = resume_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, resume_path)

    # If a specific tag is given, also save a named copy
    if tag is not None:
        tag_path = os.path.join(output_dir, f"dflash_draft_{tag}.pt")
        torch.save(checkpoint, tag_path)
        print(f"  Checkpoint saved: {tag_path}")

    return resume_path


def load_full_checkpoint(output_dir, draft_model, optimizer, scheduler, device):
    """
    Load full training state from the resume checkpoint.

    Returns:
        (epoch, batch_idx, global_step, best_loss) if checkpoint found
        None if no checkpoint exists
    """
    resume_path = os.path.join(output_dir, RESUME_CHECKPOINT_NAME)
    if not os.path.exists(resume_path):
        return None

    print(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, weights_only=False, map_location="cpu")

    # Restore model weights
    for name, param in draft_model.named_parameters():
        if name in checkpoint["model_state_dict"]:
            param.data.copy_(checkpoint["model_state_dict"][name].to(param.device))

    # Restore optimizer (handles momentum buffers, exp_avg, etc.)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Move optimizer state tensors to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Restore scheduler
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states for reproducibility
    torch.random.set_rng_state(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

    epoch = checkpoint["epoch"]
    batch_idx = checkpoint["batch_idx"]
    global_step = checkpoint["global_step"]
    best_loss = checkpoint["best_loss"]

    print(f"  Resumed at epoch {epoch+1}, batch {batch_idx}, global_step {global_step}, best_loss {best_loss:.4f}")
    return epoch, batch_idx, global_step, best_loss


def _save_model_only(draft_model, draft_config, output_dir, tag, loss):
    """Save only model weights (lightweight, for inference export)."""
    save_path = os.path.join(output_dir, f"dflash_draft_{tag}.pt")
    state_dict = {}
    for name, param in draft_model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.cpu()
    torch.save({"state_dict": state_dict, "config": draft_config, "loss": loss}, save_path)
    print(f"  Model-only checkpoint saved: {save_path}")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    global _SHUTDOWN_REQUESTED

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # -----------------------------------------------------------------------
    # Load ONLY the shared embedding + lm_head from target model
    # (We don't need the full 32B model for training — injection features
    #  are pre-extracted. We only need the frozen embed/head weights.)
    # -----------------------------------------------------------------------
    print("Loading shared embedding and lm_head from target model...")
    from safetensors.torch import load_file as st_load_file
    import os as _os

    embed_shard = st_load_file(_os.path.join(args.target_model_path, "model-00001-of-00026.safetensors"))
    head_shard = st_load_file(_os.path.join(args.target_model_path, "model-00026-of-00026.safetensors"))

    shared_word_embeddings = torch.nn.Embedding(262144, 4096, padding_idx=0)
    shared_word_embeddings.weight.data.copy_(embed_shard["model.word_embeddings.weight"].to(torch.bfloat16))
    shared_word_embeddings.weight.requires_grad = False
    shared_word_embeddings = shared_word_embeddings.to(device)

    shared_lm_head = torch.nn.Linear(4096, 262144, bias=False)
    shared_lm_head.weight.data.copy_(head_shard["lm_head.weight"].to(torch.bfloat16))
    shared_lm_head.weight.requires_grad = False
    shared_lm_head = shared_lm_head.to(device)

    del embed_shard, head_shard  # free ~9GB
    print(f"Shared weights loaded to {device} (embed: {shared_word_embeddings.weight.shape}, head: {shared_lm_head.weight.shape})")

    # -----------------------------------------------------------------------
    # Create draft model
    # -----------------------------------------------------------------------
    draft_config = DFlashConfig(
        hidden_size=4096,
        num_draft_layers=args.num_draft_layers,
        num_self_attn_heads=16,
        num_self_attn_kv_heads=4,
        num_cross_attn_heads=16,
        num_cross_attn_kv_heads=4,
        head_dim=64,
        ffn_intermediate=args.ffn_intermediate,
        block_size=args.block_size,
        rope_theta=8_000_000.0,
        vocab_size=262144,
    )

    draft_model = DFlashDraftModel(draft_config)

    # Directly assign shared frozen weights (no full target model needed)
    draft_model.word_embeddings = shared_word_embeddings
    draft_model.lm_head = shared_lm_head

    # Move trainable parts to device
    draft_model.kv_fusion = draft_model.kv_fusion.to(device, dtype=torch.bfloat16)
    draft_model.layers = draft_model.layers.to(device, dtype=torch.bfloat16)
    draft_model.norm = draft_model.norm.to(device, dtype=torch.bfloat16)
    draft_model.rotary_emb = draft_model.rotary_emb.to(device)

    trainable_params = draft_model.get_trainable_parameters()
    num_trainable = draft_model.count_trainable_parameters()
    num_total = draft_model.count_total_parameters()
    print(f"Draft model: {num_trainable / 1e6:.1f}M trainable / {num_total / 1e6:.1f}M total parameters")

    # -----------------------------------------------------------------------
    # Dataset and DataLoader
    # -----------------------------------------------------------------------
    print(f"Loading dataset from {args.data_dir}...")
    dataset = DFlashDataset(args.data_dir, max_seq_len=args.max_seq_len)
    print(f"Dataset: {len(dataset)} samples")

    # Use a generator with a fixed seed for reproducible shuffling across resumes
    data_generator = torch.Generator()
    data_generator.manual_seed(42)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=data_generator,
    )

    # -----------------------------------------------------------------------
    # Optimizer and Scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),  # fused only on CUDA
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # Resume from checkpoint (if exists)
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_loss = float("inf")

    resume_state = load_full_checkpoint(args.output_dir, draft_model, optimizer, scheduler, device)
    if resume_state is not None:
        start_epoch, start_batch_idx, global_step, best_loss = resume_state
        # start_batch_idx is the batch we COMPLETED last, so resume from the next one
        start_batch_idx += 1

    # -----------------------------------------------------------------------
    # Pre-compute masks and weights
    # -----------------------------------------------------------------------
    block_mask = make_block_causal_mask(
        args.max_seq_len, args.block_size, dtype=torch.bfloat16, device=device
    )
    position_weights = make_sequence_position_weights(
        args.max_seq_len, args.block_size, device=device
    )

    # -----------------------------------------------------------------------
    # Wandb
    # -----------------------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="dflash-sarvam",
                config=vars(args),
                name=f"dflash-{args.num_draft_layers}L-{args.ffn_intermediate}ffn",
                resume="allow",  # Allow wandb to resume a previous run
            )
        except ImportError:
            print("wandb not available, skipping logging")
            args.use_wandb = False

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print(f"\nStarting training: {args.epochs} epochs, {total_steps} optimizer steps", flush=True)
    print(f"  Checkpoint interval: every {args.save_interval} optimizer steps", flush=True)
    print(f"  Resume: epoch={start_epoch+1}, batch={start_batch_idx}, step={global_step}", flush=True)
    print(f"  Dataloader: {len(dataloader)} batches/epoch", flush=True)
    sys.stdout.flush()

    for epoch in range(start_epoch, args.epochs):
        draft_model.train()
        draft_model.word_embeddings.eval()
        draft_model.lm_head.eval()

        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        # Determine which batch to start from in this epoch
        skip_batches = start_batch_idx if epoch == start_epoch else 0

        for batch_idx, batch in enumerate(dataloader):
            # Skip batches we already processed (on resume)
            if batch_idx < skip_batches:
                continue

            # Check for graceful shutdown
            if _SHUTDOWN_REQUESTED:
                print(f"\nSaving checkpoint before shutdown (epoch={epoch}, batch={batch_idx})...")
                save_full_checkpoint(
                    draft_model, draft_config, optimizer, scheduler,
                    epoch, batch_idx, global_step, best_loss,
                    args.output_dir, tag=f"interrupted_e{epoch}_b{batch_idx}",
                )
                print("Checkpoint saved. Exiting gracefully.")
                if args.use_wandb:
                    wandb.finish()
                sys.exit(0)

            if batch_idx == 0 and epoch == start_epoch:
                print(f"  First batch loading...", flush=True)

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            injection_feats = batch["injection_features"].to(device)
            teacher_logit_vals = batch["teacher_top_logits"].to(device)
            teacher_logit_idxs = batch["teacher_top_indices"].to(device)

            # Trim to max_seq_len and ensure divisible by block_size
            S = input_ids.shape[1]
            S = (S // args.block_size) * args.block_size
            input_ids = input_ids[:, :S]
            injection_feats = injection_feats[:, :S]
            teacher_logit_vals = teacher_logit_vals[:, :S]
            teacher_logit_idxs = teacher_logit_idxs[:, :S]

            current_mask = block_mask[:S, :S] if S < args.max_seq_len else block_mask

            # Only use autocast on CUDA; MPS and CPU run without it
            autocast_enabled = torch.cuda.is_available()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                draft_logits = draft_model(
                    input_ids=input_ids,
                    injection_features=injection_feats,
                    block_mask=current_mask,
                )

                # Position-weighted CE loss
                shift_logits = draft_logits[:, :-1]
                shift_labels = input_ids[:, 1:]
                per_token_ce = chunked_cross_entropy(shift_logits, shift_labels)
                pw = position_weights[:S - 1]
                ce_loss = (per_token_ce * pw.unsqueeze(0)).sum() / (pw.sum() * per_token_ce.shape[0])

                # Position-weighted KL distillation loss
                per_token_kl = sparse_kl_divergence(
                    draft_logits[:, :-1],
                    teacher_logit_vals[:, 1:],
                    teacher_logit_idxs[:, 1:],
                )
                kl_loss = (per_token_kl * pw.unsqueeze(0)).sum() / (pw.sum() * per_token_kl.shape[0])

                loss = (1.0 - args.kl_weight) * ce_loss + args.kl_weight * kl_loss
                loss = loss / args.grad_accum

            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Periodic checkpoint (every save_interval optimizer steps)
                if global_step % args.save_interval == 0:
                    save_full_checkpoint(
                        draft_model, draft_config, optimizer, scheduler,
                        epoch, batch_idx, global_step, best_loss,
                        args.output_dir,
                    )
                    print(f"  [Auto-save] Resume checkpoint updated at step {global_step}")

                # Logging
                if global_step % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    total_loss = loss.item() * args.grad_accum
                    elapsed = ""
                    print(
                        f"[Epoch {epoch+1}/{args.epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"Batch {batch_idx+1}/{len(dataloader)} | "
                        f"Loss: {total_loss:.4f} (CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}) | "
                        f"LR: {lr:.2e} | GradNorm: {grad_norm:.2f}",
                        flush=True,
                    )

                    if args.use_wandb:
                        wandb.log({
                            "loss": total_loss,
                            "ce_loss": ce_loss.item(),
                            "kl_loss": kl_loss.item(),
                            "lr": lr,
                            "grad_norm": grad_norm,
                            "epoch": epoch,
                            "step": global_step,
                        })

            epoch_loss += loss.item() * args.grad_accum
            epoch_ce_loss += ce_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        # End of epoch
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_ce = epoch_ce_loss / max(num_batches, 1)
        avg_kl = epoch_kl_loss / max(num_batches, 1)
        print(f"\n=== Epoch {epoch+1} complete === Avg Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KL: {avg_kl:.4f})\n")

        # Save end-of-epoch checkpoint (full state for resume + model-only for export)
        save_full_checkpoint(
            draft_model, draft_config, optimizer, scheduler,
            epoch, len(dataloader) - 1, global_step, best_loss,
            args.output_dir, tag=f"epoch_{epoch+1}",
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_model_only(draft_model, draft_config, args.output_dir, "best", avg_loss)

    # Final save
    _save_model_only(draft_model, draft_config, args.output_dir, "final", avg_loss)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints in: {args.output_dir}")

    if args.use_wandb:
        wandb.finish()


def load_draft_from_checkpoint(checkpoint_path: str, target_model) -> DFlashDraftModel:
    """Load a trained draft model from checkpoint, sharing target embeddings."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    config = ckpt["config"]

    draft_model = DFlashDraftModel(config)
    draft_model.share_target_embeddings(target_model)

    # Support both full-state and model-only checkpoints
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))

    missing = []
    for name, param in draft_model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])
        elif param.requires_grad:
            missing.append(name)

    if missing:
        print(f"Warning: {len(missing)} missing trainable keys: {missing[:5]}...")
    return draft_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash draft model for Sarvam-30B")

    # Model paths
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    # Draft model architecture
    parser.add_argument("--num_draft_layers", type=int, default=6)
    parser.add_argument("--ffn_intermediate", type=int, default=2048)
    parser.add_argument("--block_size", type=int, default=16)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--kl_weight", type=float, default=0.5)

    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save resume checkpoint every N optimizer steps (default: 50)")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
