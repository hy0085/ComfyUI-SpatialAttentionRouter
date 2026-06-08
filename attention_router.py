"""
Layer 3+4 — Attention Router Hook & Composable Patch Runtime

Integrates SpatialAttentionRouterCrossAttention into SDXL's UNet.
Uses ComfyUI's model patching system for composability with other plugins.

Key design:
- Uses set_model_attn2_replace to swap cross-attention modules
- Stores router config in model_options["transformer_options"] for shared access
- Composable with IPAdapter, PAG, HyperTile via separate hook mechanisms
- Follows DenseDiffusion's patching pattern for SD version detection
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from .cross_attention import SpatialAttentionBias, SpatialAttentionRouterCrossAttention
from .region_parser import ColorRegionParser

log = logging.getLogger("SpatialAttentionRouter")


# ---------------------------------------------------------------------------
# SD Version Detection (compatible with ComfyUI internals)
# ---------------------------------------------------------------------------

class SDVersion(Enum):
    UNKNOWN = 0
    SD1x = 1
    SD2x = 2
    SDXL = 3


class BlockType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    MIDDLE = "middle"


class TransformerID:
    """Identifies a transformer block within the UNet."""
    def __init__(
        self,
        block_type: BlockType,
        block_id: int,
        block_index: int,
        transformer_index: int,
    ):
        self.block_type = block_type
        self.block_id = block_id
        self.block_index = block_index
        self.transformer_index = transformer_index


def detect_sd_version(model) -> SDVersion:
    """Detect the Stable Diffusion version from a ModelPatcher.

    Tries multiple detection strategies for robustness across ComfyUI versions.
    """
    try:
        import comfy.model_base

        inner = model.model  # Unwrap ModelPatcher
        if isinstance(
            inner,
            (
                comfy.model_base.SDXL,
                comfy.model_base.SDXLRefiner,
                comfy.model_base.SDXL_instructpix2pix,
            ),
        ):
            return SDVersion.SDXL

        if isinstance(
            inner,
            (
                comfy.model_base.SD21,
                comfy.model_base.SD21Refiner,
            ),
        ):
            return SDVersion.SD2x

        if isinstance(
            inner,
            (
                comfy.model_base.SD15,
                comfy.model_base.SD20,
            ),
        ):
            return SDVersion.SD1x

        # Fallback: try string-based detection
        model_name = str(type(inner)).lower()
        if "sdxl" in model_name:
            return SDVersion.SDXL
        if "sd2" in model_name:
            return SDVersion.SD2x
        return SDVersion.SD1x

    except ImportError:
        log.warning("Could not import comfy.model_base, defaulting to SDXL")
        return SDVersion.SDXL
    except Exception as e:
        log.warning(f"SD version detection failed ({e}), defaulting to SDXL")
        return SDVersion.SDXL


def get_transformer_ids(sd_version: SDVersion) -> List[TransformerID]:
    """Get the list of transformer block IDs for a given SD version.

    These are the blocks that contain cross-attention and need patching.
    """
    if sd_version == SDVersion.SDXL:
        transformer_index = 0
        ids = []

        # Input blocks: 4, 5, 7, 8
        for block_id in [4, 5, 7, 8]:
            block_indices = range(2) if block_id in [4, 5] else range(10)
            for idx in block_indices:
                ids.append(TransformerID(BlockType.INPUT, block_id, idx, transformer_index))
            transformer_index += 1

        # Middle block
        for idx in range(10):
            ids.append(TransformerID(BlockType.MIDDLE, 0, idx, transformer_index))
        transformer_index += 1

        # Output blocks: 0-5
        for block_id in range(6):
            block_indices = range(2) if block_id in [3, 4, 5] else range(10)
            for idx in block_indices:
                ids.append(TransformerID(BlockType.OUTPUT, block_id, idx, transformer_index))
            transformer_index += 1

        return ids

    else:
        # SD1.x / SD2.x
        transformer_index = 0
        ids = []

        # Input blocks
        for block_id in [1, 2, 4, 5, 7, 8]:
            ids.append(TransformerID(BlockType.INPUT, block_id, 0, transformer_index))
            transformer_index += 1

        # Middle block
        ids.append(TransformerID(BlockType.MIDDLE, 0, 0, transformer_index))
        transformer_index += 1

        # Output blocks
        for block_id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
            ids.append(TransformerID(BlockType.OUTPUT, block_id, 0, transformer_index))
            transformer_index += 1

        return ids


# ---------------------------------------------------------------------------
# Router Config — stored in model_options for composability
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Configuration for the spatial attention router.

    Stored in model_options["transformer_options"]["spatial_attention_router"]
    so other plugins can inspect/modify it.
    """

    # Spatial bias instance (shared across all cross-attention blocks)
    spatial_bias: SpatialAttentionBias | None = None

    # Region masks in original resolution
    region_masks: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Token affinity ranges: [(color, start, end), ...]
    affinity_ranges: List[Tuple[str, int, int]] = field(default_factory=list)

    # Routing strength (0=off, 1=max)
    alpha: float = 0.8

    # Bias strength multiplier (higher = stronger isolation, default 8.0)
    strength: float = 12.0

    # Number of global tokens
    num_global_tokens: int = 0

    # Whether router is active
    enabled: bool = True


# ---------------------------------------------------------------------------
# Attention Router Patcher
# ---------------------------------------------------------------------------

class AttentionRouterPatcher:
    """Patches a ComfyUI model with spatial attention routing.

    Usage:
        patcher = AttentionRouterPatcher()
        model = patcher.apply(model, router_config)
    """

    # Track created attention modules so we can update them later
    _attention_modules: List[SpatialAttentionRouterCrossAttention] = []

    @classmethod
    def apply(
        cls,
        model,
        router_config: RouterConfig,
        sd_version: SDVersion | None = None,
    ) -> Tuple[Any, SpatialAttentionBias]:
        """Apply spatial attention routing to a model.

        Args:
            model: ComfyUI ModelPatcher instance.
            router_config: RouterConfig with region masks and affinity.
            sd_version: SD version (auto-detected if None).

        Returns:
            Tuple of (patched_model, spatial_bias).
        """
        work_model = model.clone()

        if sd_version is None:
            sd_version = detect_sd_version(work_model)

        log.info(f"Applying Spatial Attention Router to {sd_version.name} model")

        # Create spatial bias (shared across all blocks)
        spatial_bias = SpatialAttentionBias(
            region_masks=router_config.region_masks,
            affinity_ranges=router_config.affinity_ranges,
            alpha=router_config.alpha,
            num_global_tokens=router_config.num_global_tokens,
            strength=router_config.strength,
        )

        # Store config in model_options for other hooks to access
        work_model.model_options.setdefault("transformer_options", {})
        work_model.model_options["transformer_options"]["spatial_attention_router"] = (
            router_config
        )

        # Clear old tracking
        cls._attention_modules.clear()

        # Patch all cross-attention blocks
        # Each block gets its own attention module instance (matching DenseDiffusion pattern)
        # but they all share the same SpatialAttentionBias object
        transformer_ids = get_transformer_ids(sd_version)
        patched_count = 0

        for t_id in transformer_ids:
            try:
                module = SpatialAttentionRouterCrossAttention(spatial_bias)
                work_model.set_model_attn2_replace(
                    module,
                    block_name=t_id.block_type.value,
                    number=t_id.block_id,
                    transformer_index=t_id.block_index,
                )
                cls._attention_modules.append(module)
                patched_count += 1
            except Exception as e:
                log.debug(
                    f"Could not patch transformer {t_id.block_type.value}/{t_id.block_id}"
                    f"/{t_id.block_index}: {e}"
                )

        log.info(
            f"Patched {patched_count}/{len(transformer_ids)} cross-attention blocks"
        )

        return work_model, spatial_bias

    @classmethod
    def update_config(
        cls,
        model,
        router_config: RouterConfig,
    ):
        """Update router config on an already-patched model.

        This updates the spatial bias in all patched cross-attention modules.
        """
        work_model = model.clone()

        # Create new spatial bias
        spatial_bias = SpatialAttentionBias(
            region_masks=router_config.region_masks,
            affinity_ranges=router_config.affinity_ranges,
            alpha=router_config.alpha,
            num_global_tokens=router_config.num_global_tokens,
            strength=router_config.strength,
        )

        # Update in model_options
        work_model.model_options.setdefault("transformer_options", {})
        work_model.model_options["transformer_options"]["spatial_attention_router"] = (
            router_config
        )

        # Update bias in all tracked attention modules
        for attn_module in cls._attention_modules:
            attn_module.set_spatial_bias(spatial_bias)

        return work_model, spatial_bias


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def build_router_conditioning(
    clip_model,
    color_prompts: Dict[str, str],
    region_masks: Dict[str, torch.Tensor],
    global_prompt: str = "",
    alpha: float = 0.8,
    strength: float = 12.0,
) -> Tuple[List, RouterConfig]:
    """Build conditioning and router config from color prompts and masks.

    This is the main entry point for creating region-aware conditioning.

    Args:
        clip_model: ComfyUI CLIP model.
        color_prompts: Dict mapping color_hex → prompt_text.
        region_masks: Dict mapping color_hex → mask_tensor [H, W].
        global_prompt: Optional global prompt.
        alpha: Routing strength.

    Returns:
        Tuple of (conditioning_list, router_config).
        conditioning_list is ComfyUI-compatible: [[cond, {"pooled_output": pooled}]]
    """
    from .affinity_parser import PromptAffinityParser

    # Build affinity
    affinity_data = PromptAffinityParser.build_affinity(
        clip_model=clip_model,
        color_prompts=color_prompts,
        global_prompt=global_prompt,
    )

    # Match region masks to affinity colors
    matched_masks: Dict[str, torch.Tensor] = {}
    affinity_ranges: List[Tuple[str, int, int]] = []

    # Only keep regions that have both a mask and a prompt
    colors_with_prompts = set(color_prompts.keys())
    colors_with_masks = set(region_masks.keys())

    for color_hex, start, end in affinity_data["affinity"]:
        if color_hex in colors_with_masks:
            matched_masks[color_hex] = region_masks[color_hex]
            affinity_ranges.append((color_hex, start, end))
        else:
            log.warning(
                f"Region {color_hex} has prompt but no mask, skipping spatial routing"
            )

    # Build router config
    router_config = RouterConfig(
        region_masks=matched_masks,
        affinity_ranges=affinity_ranges,
        alpha=alpha,
        strength=strength,
        num_global_tokens=affinity_data["num_global_tokens"],
        enabled=True,
    )

    # Build conditioning
    conditioning = [
        [
            affinity_data["cond"],
            {"pooled_output": affinity_data["pooled"]},
        ]
    ]

    return conditioning, router_config
