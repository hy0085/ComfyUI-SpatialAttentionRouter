"""
ComfyUI-SpatialAttentionRouter
===============================

Spatial Attention Router for SDXL/SD1.x image generation.

Use color masks to control WHERE each prompt applies in your image.
Prevents semantic bleeding between regions — a red mask for "person"
and a green mask for "lamp" means each prompt only influences its
designated area.

Based on the Spatial Attention Router architecture:
  Layer 1 — Color Region Parser (mask → region_map)
  Layer 2 — Prompt Affinity Parser (token → region mapping)
  Layer 3 — Attention Router (cross-attention hook with spatial bias)
  Layer 4 — Composable Patch Runtime (ComfyUI hook integration)
  Layer 5 — User Experience (ComfyUI nodes)

Reference projects:
  - ComfyUI-Color-Mask-Editor (color mask creation)
  - comfyui-prompt-control / AttentionCouple (regional prompting)
  - comfyui_densediffusion (DenseDiffusion spatial control)

Quick Start:
  1. Create a color mask image (use ColorMaskEditor or any image editor)
  2. Add SpatialAttentionRouter node to your workflow
  3. Connect model, clip, and mask image
  4. Enter prompts: "#ff0000: 1girl, white hair" etc.
  5. Adjust alpha (strength) and feather (softness)

Compatibility:
  - SDXL (primary target, fully supported)
  - SD1.x / SD2.x (supported, same attention structure)
  - Flux (future, requires MMDiT adaptation)
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Convenience imports
from .attention_router import RouterConfig, AttentionRouterPatcher
from .cross_attention import SpatialAttentionBias, SpatialAttentionRouterCrossAttention
from .region_parser import ColorRegionParser
from .affinity_parser import PromptAffinityParser

__version__ = "0.1.0"
__author__ = "Spatial Attention Router Team"

# Expose custom JS frontend directory to ComfyUI
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "RouterConfig",
    "AttentionRouterPatcher",
    "SpatialAttentionBias",
    "SpatialAttentionRouterCrossAttention",
    "ColorRegionParser",
    "PromptAffinityParser",
]
