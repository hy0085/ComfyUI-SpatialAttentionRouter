"""
Layer 2 — Prompt Affinity Parser
Maps prompt tokens to their target regions.

Input:
    color_prompts: dict — {"#ff0000": "1girl, white hair", "#00ff00": "desk lamp"}
    clip_tokenizer: ComfyUI CLIP tokenizer

Output:
    token_affinity: dict[int, str] — {token_index: color_hex}
    all_embeddings: concatenated CLIP embeddings
    pooled_output: pooled CLIP output

The parser:
1. Encodes each region's prompt separately
2. Tracks which token positions belong to which region
3. Concatenates all embeddings for cross-attention
4. Optionally includes a global (base) prompt that attends everywhere
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger("SpatialAttentionRouter")
log.setLevel(logging.INFO)


class PromptAffinityParser:
    """Parses per-region prompts and builds token→region affinity mapping.

    Works with ComfyUI's CLIP model to encode prompts and track
    which embedding positions correspond to which region.
    """

    @staticmethod
    def tokenize_prompt(
        clip_model, prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Tokenize and encode a single prompt using ComfyUI CLIP.

        If the prompt contains BREAK (word-boundary), splits into segments
        and encodes each independently. This creates CLIP chunk boundaries
        that reduce cross-chunk token contamination within a single region.

        Args:
            clip_model: ComfyUI CLIP model instance.
            prompt: Text prompt string.  May contain BREAK separators.

        Returns:
            Tuple of (cond, pooled_output, extra_fields).
            cond shape: [1, n_tokens, embed_dim] (BREAK multiplies n_tokens).
        """
        segments = [s.strip() for s in re.split(r"\bBREAK\b", prompt)]
        # Keep empty chunks — "A BREAK BREAK B" → [A, "", B]

        if len(segments) == 1:
            # Fast path: no BREAK, single encode
            tokens = clip_model.tokenize(prompt)
            cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
            return cond, pooled, {"tokens": tokens}

        # BREAK path: encode each segment independently
        log.info(f"[AFFINITY] BREAK prompt (first 80 chars): '{prompt[:80]}'")
        log.info(f"[AFFINITY] segments count: {len(segments)}")
        for i, seg in enumerate(segments):
            log.info(f"[AFFINITY] Segment {i+1}: '{seg[:60]}' ({len(seg)} chars)")

        chunk_conds: List[torch.Tensor] = []
        for i, segment in enumerate(segments):
            chunk_tokens = clip_model.tokenize(segment)
            if not isinstance(chunk_tokens, dict):
                raise RuntimeError(
                    "BREAK mode incompatible with current tokenizer: "
                    f"expected dict, got {type(chunk_tokens).__name__}"
                )
            # Validate structure
            for key in chunk_tokens:
                chunks = chunk_tokens[key]
                if not isinstance(chunks, list):
                    raise RuntimeError(
                        f"Unexpected tokenizer structure for key '{key}': "
                        f"expected list, got {type(chunks).__name__}"
                    )
                if chunks and not isinstance(chunks[0], list):
                    raise RuntimeError(
                        f"BREAK mode incompatible with tokenizer key '{key}'. "
                        f"Expected nested list, got {type(chunks[0]).__name__}"
                    )
                log.info(
                    f"  [BREAK] Segment {i + 1}: "
                    f"{len(chunks)} chunk(s) for key '{key}'"
                )

            chunk_cond, _ = clip_model.encode_from_tokens(
                chunk_tokens, return_pooled=True
            )
            log.info(f"[AFFINITY] Segment {i+1} cond shape: {chunk_cond.shape}")
            chunk_conds.append(chunk_cond)

        # Merge all chunk conds along token dimension
        merged_cond = torch.cat(chunk_conds, dim=1)

        # Encode full prompt once for pooled output
        _, pooled = clip_model.encode_from_tokens(
            clip_model.tokenize(prompt), return_pooled=True
        )

        log.info(f"[AFFINITY] merged_cond shape: {merged_cond.shape}")
        log.info(f"[AFFINITY] pooled shape: {pooled.shape}")
        expected_pooled_dim = 1280
        if pooled.shape[-1] != expected_pooled_dim:
            log.warning(
                f"[WARN] pooled_output dimension mismatch: "
                f"{pooled.shape[-1]} != {expected_pooled_dim} (SDXL standard)"
            )
        log.info(
            f"  [BREAK] Total {len(segments)} segments → "
            f"{merged_cond.shape[1]} tokens"
        )
        return merged_cond, pooled, {"tokens": clip_model.tokenize(prompt)}

    @staticmethod
    def build_affinity(
        clip_model,
        color_prompts: Dict[str, str],
        global_prompt: str = "",
    ) -> Dict:
        """Build token affinity mapping from color→prompt dictionary.

        Encodes each region's prompt separately, then concatenates all
        embeddings while tracking which token indices belong to which color.

        Args:
            clip_model: ComfyUI CLIP model.
            color_prompts: Dict mapping color_hex → prompt_text.
                Example: {"#ff0000": "1girl, white hair", "#00ff00": "desk lamp"}
            global_prompt: Optional global prompt that attends everywhere.
                If empty, uses empty string as base.

        Returns:
            Dict with keys:
                - "cond": Concatenated conditioning tensor [1, total_tokens, embed_dim]
                - "pooled": Pooled output tensor
                - "affinity": List of (color_hex, start_idx, end_idx) per region
                - "num_global_tokens": Number of global/base tokens
                - "region_colors": List of region color keys in order
        """
        # Encode global prompt
        global_cond, global_pooled, _ = PromptAffinityParser.tokenize_prompt(
            clip_model, global_prompt if global_prompt else ""
        )

        num_global_tokens = global_cond.shape[1]
        log.info(f"Global prompt: {num_global_tokens} tokens")
        log.info(f"[AFFINITY] global_cond shape: {global_cond.shape}")
        log.info(f"[AFFINITY] global_pooled shape: {global_pooled.shape}")

        # Encode each region's prompt
        region_conds: List[torch.Tensor] = []
        region_colors: List[str] = []
        affinity_ranges: List[Tuple[str, int, int]] = []

        current_offset = num_global_tokens

        for color_hex, prompt_text in color_prompts.items():
            if not prompt_text.strip():
                log.warning(f"Empty prompt for region {color_hex}, skipping")
                continue

            region_cond, _, _ = PromptAffinityParser.tokenize_prompt(
                clip_model, prompt_text
            )
            n_tokens = region_cond.shape[1]

            region_conds.append(region_cond)
            region_colors.append(color_hex)
            affinity_ranges.append((color_hex, current_offset, current_offset + n_tokens))

            log.info(
                f"Region {color_hex}: '{prompt_text}' → {n_tokens} tokens "
                f"(positions {current_offset}:{current_offset + n_tokens})"
            )
            current_offset += n_tokens

        # Concatenate: [global | region_1 | region_2 | ...]
        all_conds = [global_cond] + region_conds
        for i, c in enumerate(all_conds):
            log.info(f"[AFFINITY] all_conds[{i}] shape: {c.shape}")
        concat_cond = torch.cat(all_conds, dim=1)  # [1, total_tokens, embed_dim]
        log.info(f"[AFFINITY] concat_cond shape: {concat_cond.shape}")

        # Build mask: for each region's tokens, a binary indicator
        total_tokens = concat_cond.shape[1]
        token_to_region: Dict[int, str] = {}

        for color_hex, start, end in affinity_ranges:
            for t in range(start, end):
                token_to_region[t] = color_hex

        log.info(f"[AFFINITY] concat_cond final shape: {concat_cond.shape}")
        log.info(f"[AFFINITY] global_pooled final shape: {global_pooled.shape}")
        log.info(
            f"Total: {total_tokens} tokens, "
            f"{num_global_tokens} global + {sum(c.shape[1] for c in region_conds)} regional "
            f"across {len(region_colors)} regions"
        )

        return {
            "cond": concat_cond,
            "pooled": global_pooled,
            "affinity": affinity_ranges,  # [(color, start, end), ...]
            "token_to_region": token_to_region,  # {token_idx: color}
            "num_global_tokens": num_global_tokens,
            "region_colors": region_colors,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def build_affinity_from_conditionings(
        base_cond: torch.Tensor,
        base_pooled: torch.Tensor,
        region_conds: List[Tuple[str, torch.Tensor]],
    ) -> Dict:
        """Build affinity from pre-encoded conditionings.

        Use this when conditionings come from other ComfyUI nodes.

        Args:
            base_cond: Base/global conditioning [1, n_tokens, embed_dim].
            base_pooled: Base pooled output.
            region_conds: List of (color_hex, cond_tensor) for each region.

        Returns:
            Same dict format as build_affinity().
        """
        log.info("========== AFFINITY DEBUG ==========")
        log.info(
            f"global_cond.shape: "
            f"{base_cond.shape if base_cond is not None else 'None'}"
        )
        log.info(
            f"global_pooled.shape: "
            f"{base_pooled.shape if base_pooled is not None else 'None'}"
        )

        num_global_tokens = base_cond.shape[1]
        affinity_ranges = []
        region_colors = []
        region_tensors = []
        token_to_region: Dict[int, str] = {}

        current_offset = num_global_tokens

        for color_hex, cond in region_conds:
            n_tokens = cond.shape[1]
            log.info(
                f"region_cond '{color_hex}' shape: "
                f"{cond.shape if cond is not None else 'None'}"
            )
            region_tensors.append(cond)
            region_colors.append(color_hex)
            affinity_ranges.append((color_hex, current_offset, current_offset + n_tokens))
            for t in range(current_offset, current_offset + n_tokens):
                token_to_region[t] = color_hex
            current_offset += n_tokens

        concat_cond = torch.cat([base_cond] + region_tensors, dim=1)
        log.info(
            f"concat_cond.shape: "
            f"{concat_cond.shape if concat_cond is not None else 'None'}"
        )
        log.info("====================================")

        return {
            "cond": concat_cond,
            "pooled": base_pooled,
            "affinity": affinity_ranges,
            "token_to_region": token_to_region,
            "num_global_tokens": num_global_tokens,
            "region_colors": region_colors,
            "total_tokens": concat_cond.shape[1],
        }
