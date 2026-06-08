"""
Layer 3 — Spatial Attention Router (v2.1-beta: Output Blending + Region Logit Bias)

Architecture:
    1. KV-Isolation: Physical isolation of region tokens.
    2. Query Steering (Logit Bias): Pre-softmax spatial constraints to force token placement.
    3. Output Blending: Smooth integration via mask weighting.

Production Hardening:
    - Safe aspect-ratio inference & (H, W) based Mask Pyramid Cache.
    - Bulletproof CFG index handling for ComfyUI 3rd-party nodes (PAG, ControlNet).
    - Corrected feathered overlap math for global background mask.
    - NaN-prevention for zero-length token regions.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger("SpatialAttentionRouter")

# Debug log file setup
_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "_debug_router.log")


def _debug_log(msg: str):
    """Append a message to the debug log file."""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Spatial Attention Bias (lightweight config + mask pyramid cache)
# ---------------------------------------------------------------------------

class SpatialAttentionBias:
    """Stores spatial masks, affinity, and a resolution-keyed mask cache."""

    def __init__(
        self,
        region_masks: Dict[str, torch.Tensor],
        affinity_ranges: List[Tuple[str, int, int]],
        alpha: float = 0.9,          # preserved for interface compatibility
        num_global_tokens: int = 0,
        strength: float = 12.0,
    ):
        self.region_masks = region_masks
        self.affinity_ranges = affinity_ranges
        self.num_global_tokens = num_global_tokens
        self.strength = strength

        # Mask pyramid cache indexed by (latent_h, latent_w).
        # Prevents cache-key collisions: 64×64 ≠ 32×128 even though both
        # have 4096 spatial positions.
        self._mask_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        _debug_log(
            f"SpatialAttentionBias INIT (v2.1-beta): "
            f"regions={list(region_masks.keys())}, "
            f"affinity_ranges={[(c, s, e) for c, s, e in affinity_ranges]}, "
            f"num_global_tokens={num_global_tokens}, strength={strength}"
        )


# ---------------------------------------------------------------------------
# Cross-Attention Module (Output Blending + Query Steering)
# ---------------------------------------------------------------------------

class SpatialAttentionRouterCrossAttention(torch.nn.Module):
    """v2.1-beta: Output Blending with Region Logit Bias steering.

    Replaces the standard CrossAttention in SDXL's BasicTransformerBlock.

    Key changes over v2.0:
      - Mask cache keyed by (H, W) instead of spatial_len.
      - sum_masks accumulator → correct global mask in feathered overlap zones.
      - start >= end guard → no NaN from zero-length token regions.
      - Safe CFG: batch index bounds-check + diagnostic logging on first call.
      - Hybrid steering: Logit Bias on region tokens within each isolated pass.
    """

    _logged_cfg_state = False

    def __init__(self, spatial_bias: SpatialAttentionBias | None = None):
        super().__init__()
        self.spatial_bias = spatial_bias

    def set_spatial_bias(self, bias: SpatialAttentionBias):
        self.spatial_bias = bias

    def _get_resized_masks(
        self,
        spatial_len: int,
        extra_options: dict,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Resize all region masks to the current UNet block resolution.

        Cache key is (latent_h, latent_w) to prevent shape collisions.
        """
        # Prefer original_shape for aspect ratio (set by ComfyUI on initial latent)
        orig_shape = extra_options.get("original_shape", None)
        if orig_shape is not None and len(orig_shape) >= 4:
            aspect_ratio = orig_shape[2] / orig_shape[3]
        else:
            orig_h, orig_w = next(iter(self.spatial_bias.region_masks.values())).shape
            aspect_ratio = orig_h / orig_w

        latent_w = int(round(math.sqrt(spatial_len / aspect_ratio)))
        latent_h = int(round(latent_w * aspect_ratio))

        if latent_h * latent_w != spatial_len:
            latent_h = spatial_len // latent_w

        cache_key = (latent_h, latent_w)

        if cache_key in self.spatial_bias._mask_cache:
            return self.spatial_bias._mask_cache[cache_key]

        masks_4d: Dict[str, torch.Tensor] = {}
        for color, mask_2d in self.spatial_bias.region_masks.items():
            m = mask_2d.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
            m = F.interpolate(m, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
            masks_4d[color] = m.view(1, 1, spatial_len, 1)

        self.spatial_bias._mask_cache[cache_key] = masks_4d
        return masks_4d

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        extra_options: dict,
    ):
        """Output-blended + logit-bias-steered cross-attention forward.

        Called by ComfyUI's BasicTransformerBlock:
            q = self.attn2.to_q(n)         [B, H*W, heads*dim_head]
            k = self.attn2.to_k(context)    [B, N_tokens, heads*dim_head]
            v = self.attn2.to_v(context)    [B, N_tokens, heads*dim_head]

        Returns:
            Attention output [B, H*W, heads*dim_head]
        """
        heads: int = extra_options["n_heads"]
        b, _, dim_head_total = q.shape
        dim_head = dim_head_total // heads

        # Reshape: [B, N, heads, dim_head] → [B, heads, N, dim_head]
        q = q.view(b, -1, heads, dim_head).transpose(1, 2)
        k = k.view(b, -1, heads, dim_head).transpose(1, 2)
        v = v.view(b, -1, heads, dim_head).transpose(1, 2)

        spatial_len = q.shape[2]  # H*W
        scale = 1.0 / math.sqrt(dim_head)

        # Standard SDPA fallback
        def calc_attn_standard(q_x, k_x, v_x):
            if hasattr(F, 'scaled_dot_product_attention'):
                return F.scaled_dot_product_attention(q_x, k_x, v_x, dropout_p=0.0)
            return torch.softmax(q_x @ k_x.transpose(-2, -1) * scale, dim=-1) @ v_x

        # 1. Fallback: router disabled or no regions defined
        if self.spatial_bias is None or not self.spatial_bias.affinity_ranges:
            out = calc_attn_standard(q, k, v)
            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        masks_4d = self._get_resized_masks(spatial_len, extra_options, q.device, q.dtype)
        out_accum = torch.zeros_like(q)
        num_global = self.spatial_bias.num_global_tokens

        # Diagnostic logging on first forward — surfaces CFG / plugin interactions
        cond_or_uncond = extra_options.get("cond_or_uncond", [])
        if not SpatialAttentionRouterCrossAttention._logged_cfg_state:
            _debug_log("--- INIT V2.1-BETA DIAGNOSTICS ---")
            _debug_log(f"Extra Options Keys: {list(extra_options.keys())}")
            _debug_log(f"Batch Size: {b}, CFG Array: {cond_or_uncond}")
            SpatialAttentionRouterCrossAttention._logged_cfg_state = True

        # 2. Per-batch output blending with CFG isolation
        for batch_idx in range(b):
            # Safe uncond detection — survives malformed cond_or_uncond
            is_uncond = False
            if batch_idx < len(cond_or_uncond):
                is_uncond = (cond_or_uncond[batch_idx] == 1)

            q_b = q[batch_idx:batch_idx + 1]
            k_b = k[batch_idx:batch_idx + 1]
            v_b = v[batch_idx:batch_idx + 1]

            # Negative prompt: full global attention, no masking
            if is_uncond:
                out_accum[batch_idx:batch_idx + 1] = calc_attn_standard(q_b, k_b, v_b)
                continue

            # —— Positive Prompt Routing ——
            sum_masks = torch.zeros(1, 1, spatial_len, 1, device=q.device, dtype=q.dtype)
            total_mask_weight = torch.zeros(1, 1, spatial_len, 1, device=q.device, dtype=q.dtype)

            # Slice base (global) tokens
            if num_global > 0:
                k_base = k_b[:, :, :num_global, :]
                v_base = v_b[:, :, :num_global, :]
            else:
                k_base = torch.empty(1, heads, 0, dim_head, device=q.device, dtype=q.dtype)
                v_base = torch.empty(1, heads, 0, dim_head, device=q.device, dtype=q.dtype)

            # Process each isolated region
            for color, start, end in self.spatial_bias.affinity_ranges:
                # NaN shield: skip empty (zero-length) token ranges
                if start >= end:
                    continue

                if color not in masks_4d:
                    continue

                m = masks_4d[color]          # [1, 1, spatial_len, 1]

                if not hasattr(self, "_logged_router_hit"):
                    self._logged_router_hit = True
                    print(
                        f"[ROUTER] region={color} "
                        f"tokens={start}:{end} "
                        f"mask_pixels={(m > 0.5).sum().item()}"
                    )

                sum_masks += m
                total_mask_weight += m

                k_reg = k_b[:, :, start:end, :]
                v_reg = v_b[:, :, start:end, :]

                # ADDCOMM: combine base (quality/style) with region (specific)
                if num_global > 0:
                    k_combined = torch.cat([k_base, k_reg], dim=2)
                    v_combined = torch.cat([v_base, v_reg], dim=2)
                else:
                    k_combined = k_reg
                    v_combined = v_reg

                # — Region Logit Bias Injection —
                attn_logits = q_b @ k_combined.transpose(-2, -1) * scale

                # --- Diagnostic probe (first batch only) ---
                if not hasattr(self, "_logged_tensor_stats"):
                    _debug_log(
                        f"Attention Tensor Stats — "
                        f"mask min={m.min().item():.3f} max={m.max().item():.3f} "
                        f"mean={m.mean().item():.3f} | "
                        f"logits min={attn_logits.min().item():.3f} "
                        f"max={attn_logits.max().item():.3f} "
                        f"mean={attn_logits.mean().item():.3f} "
                        f"std={attn_logits.std().item():.3f}"
                    )
                    self._logged_tensor_stats = True

                # --- Quadratic decay (replaces linear penalty) ---
                # bias = -strength * (1 - m)²
                #   m=1.0 (core)  → bias = 0        (no penalty)
                #   m=0.8 (feather)→ bias = -0.04*S  (gentle, preserves soft edge)
                #   m=0.0 (bg)    → bias = -1.0*S    (max isolation)
                steering_bias = (
                    -self.spatial_bias.strength * ((1.0 - m) ** 2)
                )

                if not hasattr(self, "_logged_bias_stats"):
                    _debug_log(
                        f"Bias Applied — "
                        f"min={steering_bias.min().item():.3f} "
                        f"max={steering_bias.max().item():.3f}"
                    )
                    self._logged_bias_stats = True

                if num_global > 0:
                    attn_logits[:, :, :, num_global:] += steering_bias
                else:
                    attn_logits += steering_bias

                attn_weight = torch.softmax(attn_logits, dim=-1)
                out_reg = attn_weight @ v_combined

                out_accum[batch_idx:batch_idx + 1] += out_reg * m

            # Compute background mask: pixels not covered by any region.
            # Accumulating sum_masks first then subtracting handles feathering
            # overlaps correctly — no negative clamping artifacts.
            global_mask = torch.clamp(1.0 - sum_masks, min=0.0)
            total_mask_weight += global_mask
            total_mask_weight = torch.clamp(total_mask_weight, min=1e-6)

            if num_global > 0:
                out_bg = calc_attn_standard(q_b, k_base, v_base)
                out_accum[batch_idx:batch_idx + 1] += out_bg * global_mask

            # Normalize — weighted average handles feathering overlap zones
            out_accum[batch_idx:batch_idx + 1] /= total_mask_weight

        # Reshape back: [B, heads, H*W, dim_head] → [B, H*W, heads*dim_head]
        return out_accum.transpose(1, 2).reshape(b, -1, heads * dim_head)
