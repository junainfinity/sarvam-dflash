"""
Sarvam-30B wrapper for DFlash KV injection feature extraction.

Subclasses SarvamMoEForCausalLM to expose hidden-state extraction from
5 uniformly sampled layers and optional teacher logit extraction for
white-box distillation.
"""

import sys
import os
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# Import from sarvam-30b as a package
_sarvam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sarvam-30b")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure sarvam-30b can be imported as a package
import importlib
import importlib.util

def _import_sarvam_module(name):
    """Import a module from sarvam-30b directory, handling relative imports."""
    spec = importlib.util.spec_from_file_location(
        f"sarvam_30b.{name}",
        os.path.join(_sarvam_dir, f"{name}.py"),
        submodule_search_locations=[_sarvam_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"sarvam_30b.{name}"] = module
    return spec, module

# First import config (no deps)
_cfg_spec, _cfg_mod = _import_sarvam_module("configuration_sarvam_moe")
_cfg_spec.loader.exec_module(_cfg_mod)
SarvamMoEConfig = _cfg_mod.SarvamMoEConfig

# Patch relative import: modeling_sarvam_moe imports .configuration_sarvam_moe
# We need to make it available as a relative import
import types
_fake_pkg = types.ModuleType("sarvam_30b")
_fake_pkg.__path__ = [_sarvam_dir]
_fake_pkg.__package__ = "sarvam_30b"
_fake_pkg.configuration_sarvam_moe = _cfg_mod
sys.modules["sarvam_30b"] = _fake_pkg

# Now import the modeling file
_model_spec = importlib.util.spec_from_file_location(
    "sarvam_30b.modeling_sarvam_moe",
    os.path.join(_sarvam_dir, "modeling_sarvam_moe.py"),
    submodule_search_locations=[_sarvam_dir],
)
_model_mod = importlib.util.module_from_spec(_model_spec)
# Set the package so relative imports work
_model_mod.__package__ = "sarvam_30b"
sys.modules["sarvam_30b.modeling_sarvam_moe"] = _model_mod
_model_spec.loader.exec_module(_model_mod)

SarvamMoEForCausalLM = _model_mod.SarvamMoEForCausalLM


# 5 uniformly spaced layer indices from 19 layers (0-indexed into decoder layers)
# hidden_states tuple: index 0 = embedding output, index i+1 = layer i output
# So we pick hidden_states[1], [5], [10], [14], [19]
KV_INJECTION_LAYER_INDICES = [0, 4, 9, 13, 18]
KV_INJECTION_HIDDEN_STATE_INDICES = [i + 1 for i in KV_INJECTION_LAYER_INDICES]
NUM_INJECTION_LAYERS = len(KV_INJECTION_LAYER_INDICES)


class SarvamMoEForCausalLMWithKVInjection(SarvamMoEForCausalLM):
    """Sarvam-30B with methods to extract KV injection features and teacher logits."""

    @torch.no_grad()
    def get_kv_injection_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract hidden states from 5 uniformly sampled layers and concatenate.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len] optional

        Returns:
            injection_features: [B, seq_len, 20480] (5 * 4096)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # outputs.hidden_states: tuple of 20 tensors, each [B, seq_len, 4096]
        selected = [outputs.hidden_states[i] for i in KV_INJECTION_HIDDEN_STATE_INDICES]
        return torch.cat(selected, dim=-1)  # [B, seq_len, 20480]

    @torch.no_grad()
    def get_teacher_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k teacher logits for white-box distillation (memory efficient).

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len] optional
            top_k: number of top logits to retain

        Returns:
            top_logits: [B, seq_len, top_k] float32
            top_indices: [B, seq_len, top_k] int64
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()  # [B, S, V]
        top_logits, top_indices = logits.topk(top_k, dim=-1)  # [B, S, k], [B, S, k]
        return top_logits, top_indices

    @torch.no_grad()
    def get_injection_and_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass returning both injection features and teacher logits.

        Returns:
            injection_features: [B, seq_len, 20480]
            top_logits: [B, seq_len, top_k]
            top_indices: [B, seq_len, top_k]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # Injection features
        selected = [outputs.hidden_states[i] for i in KV_INJECTION_HIDDEN_STATE_INDICES]
        injection_features = torch.cat(selected, dim=-1)  # [B, S, 20480]

        # Teacher logits (top-k only)
        logits = self.lm_head(outputs.last_hidden_state).float()
        top_logits, top_indices = logits.topk(top_k, dim=-1)

        return injection_features, top_logits, top_indices
