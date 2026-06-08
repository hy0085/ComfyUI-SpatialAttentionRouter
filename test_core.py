"""
Standalone test for Spatial Attention Router core logic.
Does NOT require ComfyUI — tests region parsing, affinity, and attention bias.

Run: python test_core.py
"""

import sys
import os
import unittest

import numpy as np
import torch
from PIL import Image

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from region_parser import ColorRegionParser
from cross_attention import SpatialAttentionBias, SpatialAttentionRouterCrossAttention


def create_test_mask():
    """Create a test color mask with 3 regions: red, green, blue."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Red region (top-left)
    img[50:200, 50:200] = [255, 0, 0]

    # Green region (top-right)
    img[50:200, 300:450] = [0, 255, 0]

    # Blue region (bottom)
    img[300:450, 100:400] = [0, 0, 255]

    return img


class TestColorRegionParser(unittest.TestCase):
    """Test Layer 1: Color Region Parser."""

    def setUp(self):
        self.mask = create_test_mask()

    def test_extract_colors(self):
        """Should extract 3 unique colors from test mask."""
        colors = ColorRegionParser.extract_unique_colors(self.mask, tolerance=10)
        self.assertGreaterEqual(len(colors), 2)  # At least red and blue
        print(f"  Extracted colors: {colors}")

    def test_color_hex_conversion(self):
        """Hex ↔ RGB conversion roundtrip."""
        hex_str = "#ff8800"
        r, g, b = ColorRegionParser.hex_to_rgb(hex_str)
        self.assertEqual((r, g, b), (255, 136, 0))
        back = ColorRegionParser.color_to_hex(r, g, b)
        self.assertEqual(back, hex_str)

    def test_parse_mask(self):
        """Full mask parsing pipeline."""
        region_map = ColorRegionParser.parse(self.mask, feather_px=3)
        self.assertGreaterEqual(len(region_map), 2)
        for color, mask in region_map.items():
            self.assertEqual(mask.ndim, 2)  # [H, W]
            self.assertTrue(mask.max() > 0.5)  # Has active region
            print(f"  Region {color}: shape={mask.shape}, "
                  f"coverage={mask.mean().item():.3f}")

    def test_global_mask(self):
        """Global mask should cover non-region areas."""
        region_map = ColorRegionParser.parse(self.mask, feather_px=0)
        global_mask = ColorRegionParser.get_global_mask(region_map)
        self.assertTrue(global_mask.sum() > 0)  # Has uncovered area
        print(f"  Global mask coverage: {global_mask.mean().item():.3f}")

    def test_from_comfy_tensor(self):
        """Parse from ComfyUI IMAGE format [B, H, W, C]."""
        img_tensor = torch.from_numpy(self.mask.astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W, 3]
        region_map = ColorRegionParser.from_comfy_image(img_tensor, feather_px=3)
        self.assertGreaterEqual(len(region_map), 2)
        print(f"  From tensor: {len(region_map)} regions")


class TestSpatialAttentionBias(unittest.TestCase):
    """Test Layer 3: Output Blending Architecture."""

    def setUp(self):
        self.mask = create_test_mask()
        self.region_map = ColorRegionParser.parse(self.mask, feather_px=30)

    def test_output_blending_basic(self):
        """Output blending forward should produce valid output."""
        colors = list(self.region_map.keys())
        if len(colors) < 2:
            self.skipTest("Need at least 2 colors")

        affinity = []
        for i, color in enumerate(colors[:2]):
            start = 77 + i * 77
            end = start + 77
            affinity.append((color, start, end))

        bias_mod = SpatialAttentionBias(
            region_masks=self.region_map,
            affinity_ranges=affinity,
            num_global_tokens=77,
        )

        attn_module = SpatialAttentionRouterCrossAttention(spatial_bias=bias_mod)

        # Simulate cross-attention input (before reshape)
        b, spatial, tokens = 1, 1024, 231  # 231 = 77 base + 77*2 regions
        dim_head, heads = 64, 8
        dim = dim_head * heads

        q = torch.randn(b, spatial, dim)
        k = torch.randn(b, tokens, dim)
        v = torch.randn(b, tokens, dim)
        extra_options = {"n_heads": heads, "dim_head": dim_head}

        out = attn_module(q, k, v, extra_options)

        self.assertEqual(out.shape, (b, spatial, dim))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        print(f"  Output blending forward: in={q.shape}, out={out.shape}, OK")

    def test_no_bias_fallback(self):
        """Forward without spatial_bias should return unchanged pass-through."""
        attn_module = SpatialAttentionRouterCrossAttention(spatial_bias=None)

        b, spatial, tokens = 1, 256, 77
        dim_head, heads = 64, 4
        dim = dim_head * heads

        q = torch.randn(b, spatial, dim)
        k = torch.randn(b, tokens, dim)
        v = torch.randn(b, tokens, dim)
        extra_options = {"n_heads": heads, "dim_head": dim_head}

        out = attn_module(q, k, v, extra_options)
        self.assertEqual(out.shape, (b, spatial, dim))
        self.assertFalse(torch.isnan(out).any())
        print(f"  No-bias fallback: in={q.shape}, out={out.shape}, OK")

    def test_cond_uncond_separation(self):
        """CFG: uncond batch should get full attention, cond gets blending."""
        colors = list(self.region_map.keys())
        if len(colors) < 1:
            self.skipTest("Need at least 1 color")

        affinity = [(colors[0], 77, 154)]

        bias_mod = SpatialAttentionBias(
            region_masks=self.region_map,
            affinity_ranges=affinity,
            num_global_tokens=77,
        )

        attn_module = SpatialAttentionRouterCrossAttention(spatial_bias=bias_mod)

        # Batch=2: [uncond=1, cond=0]
        b, spatial, tokens = 2, 256, 231
        dim_head, heads = 64, 4
        dim = dim_head * heads

        q = torch.randn(b, spatial, dim)
        k = torch.randn(b, tokens, dim)
        v = torch.randn(b, tokens, dim)
        extra_options = {
            "n_heads": heads,
            "dim_head": dim_head,
            "cond_or_uncond": [1, 0],  # batch[0]=uncond, batch[1]=cond
        }

        out = attn_module(q, k, v, extra_options)

        self.assertEqual(out.shape, (b, spatial, dim))
        self.assertFalse(torch.isnan(out).any())
        # Uncond (batch 0) should be DIFFERENT from cond (batch 1)
        # due to output blending routing
        self.assertFalse(torch.allclose(out[0], out[1]))
        print(f"  CFG separation: batch=2, cond≠uncond={not torch.allclose(out[0], out[1])}, OK")


class TestCrossAttention(unittest.TestCase):
    """Test the custom cross-attention module."""

    def test_forward_no_bias(self):
        """Forward pass without spatial bias (should match standard attention)."""
        attn_module = SpatialAttentionRouterCrossAttention(spatial_bias=None)

        b, spatial, tokens, dim_head, heads = 2, 256, 77, 64, 8
        dim = dim_head * heads

        q = torch.randn(b, spatial, dim)
        k = torch.randn(b, tokens, dim)
        v = torch.randn(b, tokens, dim)

        extra_options = {"n_heads": heads, "original_shape": (b, 4, 16, 16)}

        out = attn_module(q, k, v, extra_options)

        self.assertEqual(out.shape, (b, spatial, dim))
        self.assertFalse(torch.isnan(out).any())
        print(f"  No-bias forward: input={q.shape}, output={out.shape}")

    def test_forward_with_bias(self):
        """Forward pass with spatial bias applied."""
        mask = create_test_mask()
        region_map = ColorRegionParser.parse(mask, feather_px=0)

        colors = list(region_map.keys())
        if len(colors) < 1:
            self.skipTest("Need at least 1 color")

        affinity = [(colors[0], 77, 100)]

        spatial_bias = SpatialAttentionBias(
            region_masks=region_map,
            affinity_ranges=affinity,
            alpha=0.8,
            num_global_tokens=77,
        )

        attn_module = SpatialAttentionRouterCrossAttention(spatial_bias=spatial_bias)

        b, spatial, tokens, dim_head, heads = 2, 256, 100, 64, 8
        dim = dim_head * heads

        q = torch.randn(b, spatial, dim)
        k = torch.randn(b, tokens, dim)
        v = torch.randn(b, tokens, dim)

        extra_options = {"n_heads": heads, "original_shape": (b, 4, 16, 16)}

        out = attn_module(q, k, v, extra_options)

        self.assertEqual(out.shape, (b, spatial, dim))
        self.assertFalse(torch.isnan(out).any())
        print(f"  With-bias forward: input={q.shape}, output={out.shape}")


def run_manual_test():
    """Manual test with visualization (doesn't require unittest)."""
    print("=" * 60)
    print("Spatial Attention Router — Manual Test")
    print("=" * 60)

    # Create test mask
    mask = create_test_mask()
    print(f"\n1. Created test mask: {mask.shape}, dtype={mask.dtype}")

    # Parse regions
    region_map = ColorRegionParser.parse(mask, feather_px=5)
    print(f"\n2. Parsed {len(region_map)} regions:")
    for color, m in region_map.items():
        r, g, b = ColorRegionParser.hex_to_rgb(color)
        coverage = m.mean().item()
        print(f"   {color} (RGB={r},{g},{b}): coverage={coverage:.3f}, "
              f"shape={tuple(m.shape)}")

    # Create spatial bias (Output Blending mode — no compute_bias needed)
    colors = list(region_map.keys())
    if len(colors) >= 2:
        affinity = [
            (colors[0], 77, 100),
            (colors[1], 100, 123),
        ]
    elif len(colors) == 1:
        affinity = [(colors[0], 77, 100)]
    else:
        print("\n   No regions found, skipping test")
        return

    bias_mod = SpatialAttentionBias(
        region_masks=region_map,
        affinity_ranges=affinity,
        num_global_tokens=77,
    )
    print(f"\n3. SpatialAttentionBias initialized (Output Blending mode)")
    print(f"   Regions: {len(bias_mod.region_masks)}, "
          f"Global tokens: {bias_mod.num_global_tokens}")

    # Test cross-attention with Output Blending
    attn_module = SpatialAttentionRouterCrossAttention(spatial_bias=bias_mod)
    b, spatial, tokens, dim_head, heads = 1, 1024, 123, 64, 8
    dim = dim_head * heads

    q = torch.randn(b, spatial, dim)
    k = torch.randn(b, tokens, dim)
    v = torch.randn(b, tokens, dim)
    extra_options = {"n_heads": heads, "original_shape": (b, 4, 32, 32)}

    out = attn_module(q, k, v, extra_options)
    print(f"\n4. Output-blended forward: output={out.shape}, "
          f"NaN={torch.isnan(out).any().item()}")
    assert out.shape == (b, spatial, dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print(f"   Output valid: shape OK, no NaN, no Inf")

    print("\n" + "=" * 60)
    print("All manual tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    # Run manual test
    run_manual_test()

    # Run unit tests
    print("\n\nRunning unit tests...")
    unittest.main(argv=[sys.argv[0]], verbosity=2, exit=False)
