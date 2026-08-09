# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import (
    LayerNorm,
    text_global_pool,
    TextTransformer,
    VisionTransformer,
)
from .transformer import to_2tuple

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    attn_pooler_heads: int = 8
    no_ln_pre: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = "tok"
    use_ln_post: bool = True
    gpu_align: bool = False

    mixture_of_expert: bool = False
    num_of_expert: int = 1
    moe_layers: Optional[Tuple] = ()
    split: int = 1
    k: int = 1

    relative_pos_embed_type: str = ""
    global_layers: int = None

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    use_ln_post: bool = True
    gpu_align: bool = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = nn.GELU
    vision_heads = vision_cfg.width // vision_cfg.head_width

    norm_layer = LayerNorm

    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        no_ln_pre=vision_cfg.no_ln_pre,
        final_ln_after_pool=vision_cfg.final_ln_after_pool,
        output_dim=embed_dim,
        norm_layer=norm_layer,
        mixture_of_expert=vision_cfg.mixture_of_expert,
        num_of_expert=vision_cfg.num_of_expert,
        k=vision_cfg.k,
        moe_layers=vision_cfg.moe_layers,
        split=vision_cfg.split,
        relative_pos_embed_type=vision_cfg.relative_pos_embed_type,
        global_layers=vision_cfg.global_layers,
        use_ln_post=vision_cfg.use_ln_post,
        gpu_align=vision_cfg.gpu_align,
    )

    return visual

def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = nn.GELU

    norm_layer = LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        mlp_ratio=text_cfg.mlp_ratio,
        output_dim=embed_dim,
        norm_layer=norm_layer,
        use_ln_post=text_cfg.use_ln_post,
        gpu_align=text_cfg.gpu_align,
    )
    return text

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = torch.log(torch.tensor(1 / 0.07)).item(),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, cast_dtype=cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, cast_dtype=cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = None

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        out = features.pooled if hasattr(features, "pooled") else features
        return F.normalize(out, dim=-1) if normalize else out

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        if isinstance(x, tuple):
            x = x[0]
        x = self.ln_final(x)

        x, _ = text_global_pool(x, text)
        x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, "visual", model)
    module.image_mean = preprocess_cfg["mean"]
    module.image_std = preprocess_cfg["std"]
    module.preprocess_cfg = copy.deepcopy(
        preprocess_cfg
    )
