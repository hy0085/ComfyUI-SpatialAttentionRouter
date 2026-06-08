"""
Layer 5 — ComfyUI Node Definitions

Provides user-facing ComfyUI nodes for the Spatial Attention Router.

Nodes:
1. SpatialAttentionRouter — main all-in-one node
2. SpatialAttentionRouterAdvanced — advanced node with separate region inputs
"""

from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("SpatialAttentionRouter")
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helper: parse color prompt config string
# ---------------------------------------------------------------------------

def parse_color_prompt_config(text: str) -> Dict[str, str]:
    """Parse color→prompt config from text input.

    Supports two formats:

    1. Simple format (one per line):
        #ff0000: 1girl, white hair, standing
        #00ff00: desk lamp, glowing
        #0000ff: cyberpunk room

    2. JSON format:
        {"#ff0000": "1girl, white hair", "#00ff00": "desk lamp"}

    Args:
        text: Config text.

    Returns:
        Dict mapping color_hex → prompt_text.
    """
    text = text.strip()

    if not text:
        return {}

    # Try JSON first
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass

    # Parse line-by-line format: "#color: prompt"
    result: Dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") and ":" not in line:
            continue

        # Split on first colon
        if ":" in line:
            color_part, prompt_part = line.split(":", 1)
            color_part = color_part.strip()
            prompt_part = prompt_part.strip()

            # Validate color format
            if color_part.startswith("#") or any(
                color_part.startswith(c) for c in ("rgb", "hsl")
            ):
                result[color_part] = prompt_part
            else:
                log.warning(f"Skipping invalid color entry: {line}")
        else:
            log.warning(f"Skipping malformed line (no colon): {line}")

    return result


# ---------------------------------------------------------------------------
# Helper: build mask preview image
# ---------------------------------------------------------------------------

def _build_mask_preview(
    matched_masks: Dict[str, torch.Tensor],
    target_h: int = 512,
    target_w: int = 512,
) -> torch.Tensor:
    """Build an RGB preview image [1, H, W, 3] from matched region masks.

    Each region is rendered in its own color, overlaid on a black background.
    """
    if not matched_masks:
        return torch.zeros(1, target_h, target_w, 3)

    from .region_parser import ColorRegionParser  # local import to avoid circular deps

    preview = torch.zeros(target_h, target_w, 3)
    for color_hex, mask in matched_masks.items():
        r, g, b = ColorRegionParser.hex_to_rgb(color_hex)
        color = torch.tensor([r / 255.0, g / 255.0, b / 255.0])
        # Resize mask to target size
        m = mask.unsqueeze(0).unsqueeze(0).float()
        m = torch.nn.functional.interpolate(
            m, size=(target_h, target_w), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)  # [H, W]
        m = m.clamp(0.0, 1.0)
        for c in range(3):
            preview[:, :, c] = torch.max(preview[:, :, c], m * color[c])

    return preview.unsqueeze(0)  # [1, H, W, 3]


# ---------------------------------------------------------------------------
# Node 1: SpatialAttentionRouter (main all-in-one node)
# ---------------------------------------------------------------------------

class SpatialAttentionRouterNode:
    """Main node for Spatial Attention Router.

    Combines color mask parsing, prompt encoding, and model patching
    into a single convenient node.

    Inputs:
        - model: SDXL/SD1.x model
        - clip: CLIP model for text encoding
        - mask_image: Color-labeled mask image (IMAGE)
        - color_prompts: Color→prompt mapping text
        - global_prompt: Optional global/base prompt
        - strength: Bias multiplier (higher = stronger isolation)
        - feather_px: Soft edge feathering (pixels)

    Outputs:
        - MODEL: Patched model with spatial routing
        - CONDITIONING: Region-aware conditioning
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "color_prompts": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": (
                            "#ff0000: giant red dragon, detailed scales\n"
                            "#00ff00: tiny desk lamp, warm glow\n"
                            "#0000ff: empty blue sky, room interior"
                        ),
                    },
                ),
                "canvas_data": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "global_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Base prompt (quality/style ONLY): masterpiece, best quality, ultra detailed",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "display": "slider",
                    },
                ),
                "feather_px": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 150,
                        "step": 1,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "mask_image": ("IMAGE",),
                "base_conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("model", "conditioning", "mask_preview")
    FUNCTION = "apply"
    CATEGORY = "conditioning/颜色分区 (ColorRegion)"
    DESCRIPTION = (
        "颜色分区 (ColorRegion): 用颜色蒙版控制每句提示词生效的区域。"
        "在内置画板上涂色或连接蒙版图片。"
        "红色=人物、绿色=台灯，互不串色。"
    )

    def apply(
        self,
        model,
        clip,
        color_prompts: str = "",
        canvas_data: str = "",
        global_prompt: str = "",
        strength: float = 12.0,
        feather_px: int = 30,
        mask_image=None,
        base_conditioning=None,
    ):
        """Execute the spatial attention router node.

        Args:
            model: ComfyUI ModelPatcher.
            clip: CLIP model.
            mask_image: Color mask [B, H, W, C].
            color_prompts: Color→prompt config text.
            global_prompt: Global prompt text.
            strength:Bias multiplier — higher = stronger isolation.
            feather_px: Feather radius in pixels.
            base_conditioning: Optional pre-computed base conditioning.

        Returns:
            Tuple of (patched_model, conditioning).
        """
        from .affinity_parser import PromptAffinityParser
        from .attention_router import (
            AttentionRouterPatcher,
            RouterConfig,
            build_router_conditioning,
        )
        from .region_parser import ColorRegionParser

        # --- Step 1: Parse color prompts config ---
        color_prompt_map = parse_color_prompt_config(color_prompts)
        log.info(f"[DEBUG] color_to_prompt: {color_prompt_map}")

        if not color_prompt_map:
            log.warning("No color→prompt mappings provided, returning unmodified model")
            if base_conditioning is not None:
                log.info("[AFFINITY] fallback: returning base_conditioning as-is")
                return (model, base_conditioning, _build_mask_preview({}))
            # Encode global prompt only
            cond, pooled = clip.encode_from_tokens(
                clip.tokenize(global_prompt), return_pooled=True
            )
            log.info(f"[AFFINITY] fallback global_cond shape: {cond.shape}")
            log.info(f"[AFFINITY] fallback pooled shape: {pooled.shape}")
            return (model, [[cond, {"pooled_output": pooled}]], _build_mask_preview({}))

        log.info(f"Parsed {len(color_prompt_map)} color→prompt mappings")
        for color, prompt in color_prompt_map.items():
            log.info(f"  {color} → '{prompt}'")

        # --- Step 2: Parse color mask into region masks ---
        # Priority: in-node canvas drawing > connected mask_image
        region_masks = {}

        if canvas_data and canvas_data.startswith("data:image"):
            try:
                header, encoded = canvas_data.split(",", 1)
                img_data = base64.b64decode(encoded)
                pil_img = Image.open(BytesIO(img_data)).convert("RGB")
                region_masks = ColorRegionParser.from_pil(
                    pil_img, target_size=None, feather_px=feather_px
                )
                log.info(
                    f"Parsed {len(region_masks)} region(s) from in-node canvas"
                )
            except Exception as e:
                log.error(f"Failed to parse canvas data: {e}")

        if not region_masks and mask_image is not None:
            region_masks = ColorRegionParser.from_comfy_image(
                mask_image,
                target_size=None,
                feather_px=feather_px,
            )
            log.info(
                f"Parsed {len(region_masks)} region(s) from connected mask_image"
            )

        # Match region masks to prompts (with fuzzy color tolerance for screenshots)
        matched_masks: Dict[str, torch.Tensor] = {}
        for prompt_hex in color_prompt_map:
            prompt_rgb = ColorRegionParser.hex_to_rgb(prompt_hex)
            matched = False

            # 1. Exact match (case-insensitive)
            for mask_hex in region_masks:
                if mask_hex.lower() == prompt_hex.lower():
                    matched_masks[prompt_hex] = region_masks[mask_hex]
                    matched = True
                    break

            # 2. Fuzzy match — handles color-space drift from screenshots / JPEG
            if not matched:
                for mask_hex in region_masks:
                    mask_rgb = ColorRegionParser.hex_to_rgb(mask_hex)
                    diff = sum(abs(p - m) for p, m in zip(prompt_rgb, mask_rgb))
                    # Allow up to 45 total channel error (avg 15 per channel)
                    if diff <= 45:
                        matched_masks[prompt_hex] = region_masks[mask_hex]
                        matched = True
                        log.info(
                            f"Fuzzy matched prompt {prompt_hex} "
                            f"to mask {mask_hex} (Diff: {diff})"
                        )
                        break

            if not matched:
                log.warning(
                    f"Color {prompt_hex} not found in mask image. "
                    f"Available colors: {list(region_masks.keys())}"
                )

        if not matched_masks:
            log.warning(
                f"No matching regions found between prompts and mask. "
                f"color_prompt_map keys: {list(color_prompt_map.keys())}, "
                f"region_masks keys: {list(region_masks.keys())}"
            )

            if base_conditioning is not None:
                return (model, base_conditioning, _build_mask_preview({}))

            # Use PromptAffinityParser to encode fallback — this ensures
            # correct pooled dim (1280/768 auto-detected by CLIP) and
            # preserves BREAK parsing in global_prompt.
            fallback_text = global_prompt if global_prompt else ""
            cond, pooled, _ = PromptAffinityParser.tokenize_prompt(
                clip, fallback_text
            )

            log.info(
                f"[AFFINITY] fallback unmatched — cond shape: {cond.shape}, "
                f"pooled shape: {pooled.shape}"
            )
            return (model, [[cond, {"pooled_output": pooled}]], _build_mask_preview({}))

        log.info(f"Matched {len(matched_masks)} regions with masks")

        # --- Step 3: Encode prompts and build affinity ---
        if base_conditioning is not None:
            # Use pre-computed base conditioning
            base_cond = base_conditioning[0][0]
            base_pooled = base_conditioning[0][1].get("pooled_output", None)

            # Encode region prompts
            region_conds = []
            for color_hex, prompt_text in color_prompt_map.items():
                if color_hex not in matched_masks:
                    continue
                cond, pooled = clip.encode_from_tokens(
                    clip.tokenize(prompt_text), return_pooled=True
                )
                region_conds.append((color_hex, cond))

            affinity_data = PromptAffinityParser.build_affinity_from_conditionings(
                base_cond=base_cond,
                base_pooled=base_pooled if base_pooled is not None else torch.zeros(1, 768),
                region_conds=region_conds,
            )
        else:
            # Encode everything from scratch
            affinity_data = PromptAffinityParser.build_affinity(
                clip_model=clip,
                color_prompts={c: p for c, p in color_prompt_map.items() if c in matched_masks},
                global_prompt=global_prompt,
            )

        # --- Step 4: Build router config ---
        router_config = RouterConfig(
            region_masks=matched_masks,
            affinity_ranges=affinity_data["affinity"],
            alpha=1.0,  # unused in v2.1 Output Blending; preserved for interface compat
            strength=strength,
            num_global_tokens=affinity_data["num_global_tokens"],
            enabled=True,
        )

        # --- Step 5: Patch model ---
        patched_model, spatial_bias = AttentionRouterPatcher.apply(model, router_config)

        # --- Step 6: Build conditioning ---
        conditioning = [
            [
                affinity_data["cond"],
                {"pooled_output": affinity_data["pooled"]},
            ]
        ]

        # --- Step 7: Build mask preview image ---
        mask_preview = _build_mask_preview(matched_masks)

        log.info(
            f"SpatialAttentionRouter applied: "
            f"{len(matched_masks)} regions, strength={strength}, feather={feather_px}px"
        )

        # Verify token routing before building conditioning
        print("\n" + "=" * 100)
        print("[AFFINITY DEBUG] ROUTER HIT CHECK")
        print("=" * 100)
        for color_hex, start, end in affinity_data["affinity"]:
            mask = matched_masks.get(color_hex)
            mask_pixels = (mask > 0.5).sum().item() if mask is not None else 0
            print(
                f"[ROUTER] region={color_hex} "
                f"tokens={start}:{end} "
                f"mask_pixels={mask_pixels}"
            )

        print("\n" + "=" * 100)
        print("[AFFINITY DEBUG] FINAL OUTPUT")
        print("=" * 100)
        print("concat_cond shape =", affinity_data["cond"].shape)
        print("concat_cond dtype =", affinity_data["cond"].dtype)
        print("global_pooled shape =", affinity_data["pooled"].shape)
        print("global_pooled dtype =", affinity_data["pooled"].dtype)
        print("=" * 100 + "\n")

        return (patched_model, conditioning, mask_preview)


# ---------------------------------------------------------------------------
# Node 2: SpatialAttentionRouterAdvanced (separate encode + apply)
# ---------------------------------------------------------------------------

class SpatialAttentionRouterAdvancedNode:
    """Advanced node for Spatial Attention Router.

    Separate nodes for encoding and applying, allowing more complex
    workflows with intermediate conditioning manipulation.

    Inputs:
        - model: SDXL/SD1.x model
        - conditioning: Pre-encoded region-aware conditioning
        - mask_image: Color-labeled mask image
        - color_prompts: Color→prompt mapping text
        - alpha: Routing strength
        - feather_px: Edge feathering

    Outputs:
        - MODEL: Patched model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "mask_image": ("IMAGE",),
                "color_prompts": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "#ff0000: giant red dragon\n#00ff00: tiny desk lamp",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "display": "slider",
                    },
                ),
                "feather_px": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 150,
                        "step": 1,
                        "display": "slider",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/颜色分区 (ColorRegion)"
    DESCRIPTION = (
        "颜色分区 - 高级 (ColorRegion Advanced): 对预编码的 conditioning 应用颜色分区路由，适合高级工作流。"
    )

    def apply(
        self,
        model,
        conditioning,
        mask_image: torch.Tensor,
        color_prompts: str = "",
        strength: float = 12.0,
        feather_px: int = 30,
    ):
        """Apply spatial routing to pre-encoded conditioning.

        Args:
            model: ComfyUI ModelPatcher.
            conditioning: Pre-encoded conditioning from CLIPTextEncode or similar.
            mask_image: Color mask image.
            color_prompts: Color→prompt config.
            strength:Bias multiplier.
            feather_px: Feather radius.

        Returns:
            Tuple of (patched_model,).
        """
        from .attention_router import AttentionRouterPatcher, RouterConfig
        from .region_parser import ColorRegionParser

        # Parse color prompts
        color_prompt_map = parse_color_prompt_config(color_prompts)

        if not color_prompt_map:
            log.warning("No color→prompt mappings provided")
            return (model,)

        # Parse color mask
        region_masks = ColorRegionParser.from_comfy_image(
            mask_image, target_size=None, feather_px=feather_px
        )

        # Match masks to prompts (with fuzzy color tolerance)
        matched_masks = {}
        for prompt_hex in color_prompt_map:
            prompt_rgb = ColorRegionParser.hex_to_rgb(prompt_hex)
            matched = False

            # 1. Exact match
            for mask_hex in region_masks:
                if mask_hex.lower() == prompt_hex.lower():
                    matched_masks[prompt_hex] = region_masks[mask_hex]
                    matched = True
                    break

            # 2. Fuzzy match
            if not matched:
                for mask_hex in region_masks:
                    mask_rgb = ColorRegionParser.hex_to_rgb(mask_hex)
                    diff = sum(abs(p - m) for p, m in zip(prompt_rgb, mask_rgb))
                    if diff <= 45:
                        matched_masks[prompt_hex] = region_masks[mask_hex]
                        matched = True
                        log.info(
                            f"Fuzzy matched prompt {prompt_hex} "
                            f"to mask {mask_hex} (Diff: {diff})"
                        )
                        break

            if not matched:
                log.warning(f"Color {prompt_hex} not found in mask image")

        if not matched_masks:
            log.error("No matching regions found")
            return (model,)

        # Build simple affinity from conditioning token count
        cond_tensor = conditioning[0][0]  # [1, total_tokens, embed_dim]
        total_tokens = cond_tensor.shape[1]

        # Distribute tokens evenly among regions (approximate)
        # In advanced mode, the user is responsible for encoding regions properly
        n_regions = len(matched_masks)
        tokens_per_region = total_tokens // (n_regions + 1)  # +1 for base
        base_tokens = total_tokens - n_regions * tokens_per_region

        affinity_ranges = []
        current = base_tokens
        for color_hex in matched_masks:
            end = min(current + tokens_per_region, total_tokens)
            affinity_ranges.append((color_hex, current, end))
            current = end

        router_config = RouterConfig(
            region_masks=matched_masks,
            affinity_ranges=affinity_ranges,
            alpha=1.0,  # unused in v2.1 Output Blending; preserved for interface compat
            strength=strength,
            num_global_tokens=base_tokens,
            enabled=True,
        )

        patched_model, _ = AttentionRouterPatcher.apply(model, router_config)
        return (patched_model,)


# ---------------------------------------------------------------------------
# Node 3: ColorMaskPreview (utility for debugging)
# ---------------------------------------------------------------------------

class ColorMaskPreviewNode:
    """Preview node — shows parsed region masks as a debug image.

    Inputs:
        - mask_image: Color mask image (IMAGE)
        - feather_px: Feather radius

    Outputs:
        - IMAGE: Visualization of detected regions
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image": ("IMAGE",),
                "feather_px": (
                    "INT",
                    {"default": 5, "min": 0, "max": 50, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "conditioning/颜色分区 (ColorRegion)"
    DESCRIPTION = "预览检测到的颜色分区 (Preview detected color regions)."

    def preview(
        self,
        mask_image: torch.Tensor,
        feather_px: int = 30,
    ):
        """Generate a preview of detected regions.

        Args:
            mask_image: Input color mask [B, H, W, C].
            feather_px: Feather radius.

        Returns:
            Tuple of (preview_image,).
        """
        from .region_parser import ColorRegionParser

        h, w = mask_image.shape[1], mask_image.shape[2]

        # Parse regions
        region_masks = ColorRegionParser.from_comfy_image(
            mask_image, target_size=None, feather_px=feather_px
        )

        # Create preview: overlay colored masks
        preview = torch.zeros(h, w, 3, dtype=torch.float32)

        # Generate distinct colors for each region
        region_colors = [
            torch.tensor([1.0, 0.0, 0.0]),  # Red
            torch.tensor([0.0, 1.0, 0.0]),  # Green
            torch.tensor([0.0, 0.0, 1.0]),  # Blue
            torch.tensor([1.0, 1.0, 0.0]),  # Yellow
            torch.tensor([1.0, 0.0, 1.0]),  # Magenta
            torch.tensor([0.0, 1.0, 1.0]),  # Cyan
            torch.tensor([1.0, 0.5, 0.0]),  # Orange
            torch.tensor([0.5, 0.0, 1.0]),  # Purple
        ]

        for i, (color_hex, mask) in enumerate(region_masks.items()):
            tc = region_colors[i % len(region_colors)]
            # Resize mask to match preview size
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            # Tinted overlay
            for c in range(3):
                preview[:, :, c] += mask * tc[c] * 0.4

        # Clamp and add batch dim
        preview = torch.clamp(preview + 0.2, 0.0, 1.0)
        preview = preview.unsqueeze(0)  # [1, H, W, 3]

        return (preview,)


# ---------------------------------------------------------------------------
# Node Mappings (standard ComfyUI plugin interface)
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SpatialAttentionRouter": SpatialAttentionRouterNode,
    "SpatialAttentionRouterAdvanced": SpatialAttentionRouterAdvancedNode,
    "ColorMaskPreview": ColorMaskPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpatialAttentionRouter": "颜色分区 (ColorRegion)",
    "SpatialAttentionRouterAdvanced": "颜色分区 - 高级 (ColorRegion Advanced)",
    "ColorMaskPreview": "颜色蒙版预览 (Color Mask Preview)",
}
