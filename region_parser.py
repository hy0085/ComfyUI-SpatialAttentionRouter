"""
Layer 1 — Color Region Parser
Converts user-uploaded color mask image into internal region_map.

Input:  mask_image (PIL/numpy) — RGB color blocks for different regions
Output: region_map dict[color_hex, mask_tensor[H, W]]

Features:
- Extracts unique colors from mask image
- Creates binary masks per color
- Applies feathering (Gaussian blur) for soft edges
- Auto-resizes to latent resolution
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger("SpatialAttentionRouter")


class ColorRegionParser:
    """Extracts region masks from a color-labeled mask image.

    Each distinct color in the mask image represents a different region.
    The parser identifies unique colors, creates per-color binary masks,
    applies feathering, and optionally resizes to a target resolution.
    """

    # Color tolerance for matching — colors don't need to be exact
    COLOR_TOLERANCE: int = 35

    # Default feather radius in pixels (at original mask resolution)
    DEFAULT_FEATHER: int = 30

    @staticmethod
    def extract_unique_colors(
        image: np.ndarray, tolerance: int | None = None
    ) -> List[Tuple[int, int, int]]:
        """Extract unique colors from a mask image.

        Args:
            image: RGB image [H, W, 3] with values 0-255.
            tolerance: Max per-channel difference to merge nearby colors.

        Returns:
            List of (R, G, B) tuples sorted by frequency (most common first).
        """
        if tolerance is None:
            tolerance = ColorRegionParser.COLOR_TOLERANCE

        # Flatten to [N, 3]
        pixels = image.reshape(-1, 3).astype(np.int32)

        # Count color occurrences
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        # Filter out near-black and near-white (background / grid colors)
        def is_valid_color(rgb: np.ndarray) -> bool:
            # Skip pure black / very dark
            if rgb.sum() < 30:
                return False
            # Skip pure white / very light
            if rgb.sum() > 720:
                return False
            # Skip gray (all channels equal within tolerance)
            if abs(int(rgb[0]) - int(rgb[1])) < 5 and abs(int(rgb[1]) - int(rgb[2])) < 5:
                return False
            return True

        valid_mask = np.array([is_valid_color(c) for c in unique_colors])
        unique_colors = unique_colors[valid_mask]
        counts = counts[valid_mask]

        # Sort by frequency (most common first)
        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]

        # Merge nearby colors with Euclidean distance (handles JPEG /
        # anti-aliased edges better than per-channel Manhattan tolerance)
        merged: List[Tuple[int, int, int]] = []
        for color in unique_colors:
            color_tuple = (int(color[0]), int(color[1]), int(color[2]))
            is_duplicate = False
            for existing in merged:
                dist_sq = (
                    (color_tuple[0] - existing[0]) ** 2
                    + (color_tuple[1] - existing[1]) ** 2
                    + (color_tuple[2] - existing[2]) ** 2
                )
                if dist_sq <= tolerance**2:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(color_tuple)

        log.info(f"Extracted {len(merged)} unique region colors: {merged}")
        return merged

    @staticmethod
    def color_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB tuple to hex string like '#ff0000'."""
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex string like '#ff0000' to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    @classmethod
    def parse(
        cls,
        mask_image: np.ndarray,
        target_size: Tuple[int, int] | None = None,
        feather_px: int | None = None,
        tolerance: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Parse a color mask image into per-region binary masks.

        Args:
            mask_image: RGB image as numpy array [H, W, 3] with values 0-255.
            target_size: Optional (width, height) to resize masks to.
            feather_px: Gaussian blur radius in pixels for soft edges.
            tolerance: Color matching tolerance.

        Returns:
            Dict mapping color hex string → binary mask tensor [H, W] float32.
        """
        if feather_px is None:
            feather_px = cls.DEFAULT_FEATHER

        if tolerance is None:
            tolerance = cls.COLOR_TOLERANCE

        # Ensure uint8 format
        if mask_image.dtype != np.uint8:
            if mask_image.max() <= 1.0:
                mask_image = (mask_image * 255).astype(np.uint8)
            else:
                mask_image = mask_image.astype(np.uint8)

        # Remove alpha channel if present
        if mask_image.shape[-1] == 4:
            mask_image = mask_image[:, :, :3]
        elif len(mask_image.shape) == 2:
            mask_image = np.stack([mask_image] * 3, axis=-1)

        h, w = mask_image.shape[:2]
        colors = cls.extract_unique_colors(mask_image, tolerance)

        region_map: Dict[str, torch.Tensor] = {}

        for r, g, b in colors:
            color_hex = cls.color_to_hex(r, g, b)

            # Create binary mask for this color
            color_pixel = np.array([r, g, b], dtype=np.int32)
            diff = np.abs(mask_image.astype(np.int32) - color_pixel)
            match = (diff.sum(axis=-1) <= tolerance * 2).astype(np.float32)

            # Convert to tensor
            mask_tensor = torch.from_numpy(match)  # [H, W]

            # Apply feathering via Gaussian blur
            if feather_px > 0:
                kernel_size = feather_px * 4 + 1
                # Ensure odd kernel size
                if kernel_size % 2 == 0:
                    kernel_size += 1
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                # Create Gaussian kernel
                sigma = feather_px
                ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
                gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
                gauss = gauss / gauss.sum()
                kernel = gauss[:, None] @ gauss[None, :]
                kernel = kernel.unsqueeze(0).unsqueeze(0)

                # Apply separable-like blur (two 1D passes)
                mask_tensor = F.conv2d(
                    F.pad(mask_tensor, (kernel_size // 2,) * 4, mode="replicate"),
                    kernel,
                )
                mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0).squeeze(0).squeeze(0)

            # Resize to target size
            if target_size is not None:
                tw, th = target_size
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask_tensor = F.interpolate(
                    mask_tensor, size=(th, tw), mode="bilinear", align_corners=False
                )
                mask_tensor = mask_tensor.squeeze(0).squeeze(0)

            region_map[color_hex] = mask_tensor

        log.info(f"Parsed {len(region_map)} regions, sizes: {[m.shape for m in region_map.values()]}")
        return region_map

    @classmethod
    def from_comfy_image(
        cls,
        image_tensor: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
        feather_px: int | None = None,
        tolerance: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Parse from ComfyUI IMAGE tensor format [B, H, W, C].

        Args:
            image_tensor: ComfyUI image tensor [B, H, W, C] float32 0-1.
            target_size: Optional (width, height) to resize masks to.
            feather_px: Gaussian blur radius.
            tolerance: Color matching tolerance.

        Returns:
            Dict mapping color hex string → binary mask tensor [H, W].
        """
        # Take first batch item, convert to numpy uint8
        if image_tensor.dim() == 4:
            img = image_tensor[0].cpu().numpy()
        else:
            img = image_tensor.cpu().numpy()

        img_uint8 = (img * 255).astype(np.uint8)
        return cls.parse(img_uint8, target_size, feather_px, tolerance)

    @classmethod
    def from_pil(
        cls,
        pil_image: Image.Image,
        target_size: Tuple[int, int] | None = None,
        feather_px: int | None = None,
        tolerance: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Parse from PIL Image.

        Args:
            pil_image: PIL RGB Image.
            target_size: Optional (width, height) to resize masks to.
            feather_px: Gaussian blur radius.
            tolerance: Color matching tolerance.

        Returns:
            Dict mapping color hex string → binary mask tensor [H, W].
        """
        img = np.array(pil_image.convert("RGB"))
        return cls.parse(img, target_size, feather_px, tolerance)

    @staticmethod
    def merge_regions(
        region_map: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Merge all region masks into a single stacked tensor.

        Args:
            region_map: Dict of color → mask.

        Returns:
            Stacked tensor [num_regions, H, W].
        """
        masks = list(region_map.values())
        if not masks:
            return torch.zeros(0, 64, 64)
        return torch.stack(masks, dim=0)

    @staticmethod
    def get_global_mask(
        region_map: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the global (background/unassigned) mask.

        Areas not covered by any region get value 1.0.

        Args:
            region_map: Dict of color → mask.

        Returns:
            Global mask [H, W] where uncovered areas are 1.0.
        """
        if not region_map:
            return torch.ones(64, 64)

        stacked = ColorRegionParser.merge_regions(region_map)
        combined = stacked.sum(dim=0).clamp(0.0, 1.0)
        return 1.0 - combined
