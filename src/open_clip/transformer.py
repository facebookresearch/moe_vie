# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter

from .rope import apply_rotary_emb
import collections.abc
from itertools import repeat
from .triton.routed_ffn import routed_ffn_fwd
from .triton.routed_ffn_swiglu_inference import (
    routed_ffn_fwd as routed_ffn_fwd_swiglu_inference,
)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)
from .transform import PACKED

def text_global_pool(x, text: Optional[torch.Tensor] = None):
    pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    return pooled, tokens

@dataclass
class VisionTransformerOutput:
    pooled: Optional[torch.Tensor] = None
    latent: Optional[torch.Tensor] = None
    tokens: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    ids_restore: Optional[torch.Tensor] = None
    moe_loss: Optional[torch.Tensor] = 0

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(
        self,
        x,
        packed_num_windows=None,
        packed_end_idx=None,
    ):
        batch, _, _ = x.shape

        if packed_num_windows is None:
            packed_num_windows = [1]
        if packed_end_idx is None:
            packed_end_idx = [batch]

        def attn_fn(x_, _, num_windows):
            S, window_size, embed_dim = x_.shape
            x_ = x_.reshape(-1, window_size * num_windows, embed_dim)
            q_ = self.probe.repeat((x_.shape[0], 1, 1)).to(x_.dtype)
            attn = self.attn(q_, x_, x_, need_weights=False)[0]

            return attn

        attn_xs = torch.tensor_split(x, packed_end_idx[:-1], dim=0)
        attn = torch.cat(
            [
                attn_fn(x_pack, x_pack, num_windows)
                for (x_pack, num_windows) in zip(attn_xs, packed_num_windows)
            ],
            dim=0,
        )
        hidden_state = attn

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state

class AttentionImpl(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, relative_pos_embed_type: str = ""):
        super(AttentionImpl, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        from .rope import RotaryEmbedding
        self.rope = RotaryEmbedding(self.head_dim // 2)
        self.scale = self.head_dim ** (-0.5)

    def forward(self, query, attn_mask: Optional[torch.Tensor] = None, packed_img_idx: Optional[torch.Tensor] = None):
        batch, seq, embed_dim = query.shape

        proj = torch._C._nn.linear(query, self.in_proj_weight, self.in_proj_bias)
        proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q_, k_, v_ = proj[0], proj[1], proj[2]

        q_ = rearrange(q_, 'b s (h d) -> b h s d', h=self.num_heads)
        k_ = rearrange(k_, 'b s (h d) -> b h s d', h=self.num_heads)
        v_ = rearrange(v_, 'b s (h d) -> b s h d', h=self.num_heads)

        freqs_x = self.rope(packed_img_idx[:, :, PACKED.X] + 1)
        freqs_y = self.rope(packed_img_idx[:, :, PACKED.Y] + 1)
        freq = torch.cat([freqs_x, freqs_y], dim=-1)
        freq = freq.masked_fill(packed_img_idx[:, :, PACKED.IDX, None] < 0, 0)

        q_ = apply_rotary_emb(freq[:, None, :, :], q_)
        k_ = apply_rotary_emb(freq[:, None, :, :], k_)

        q_ = rearrange(q_, 'b h s d -> b s h d')
        k_ = rearrange(k_, 'b h s d -> b s h d')

        pad_tokens = (packed_img_idx[:, :, PACKED.IDX] == PACKED.ID_PAD_TOKEN)
        if attn_mask is None and pad_tokens.any():
            attn_mask = torch.logical_not(pad_tokens[:, None, None, :])

        flip_hs = lambda x: rearrange(x, 'b s h d -> b h s d')
        attn = torch._C._nn.scaled_dot_product_attention(flip_hs(q_), flip_hs(k_), flip_hs(v_), attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scale)
        attn = rearrange(attn, 'b h s d -> b s (h d)')

        return torch._C._nn.linear(attn, self.out_proj.weight, self.out_proj.bias)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        norm_layer: Callable = LayerNorm,
        relative_pos_embed_type: str = "",
        is_global: bool = False,
    ):
        super().__init__()
        self.is_global = is_global
        self.ln = norm_layer(d_model)
        self.relative_pos_embed_type = relative_pos_embed_type

        if self.relative_pos_embed_type:
            self.attn = AttentionImpl(
                embed_dim=d_model, num_heads=n_head, relative_pos_embed_type=relative_pos_embed_type,
            )
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ls = nn.Identity()

    def _call_attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        packed_img_idx: Optional[torch.Tensor] = None,
        tokens_per_img: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        if attn_mask is not None:
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        if self.relative_pos_embed_type:
            return self.attn(q_x, attn_mask=attn_mask, packed_img_idx=packed_img_idx), None
        else:
            return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def packed_global_attn(self, x, attn_mask, packed_img_idx, windows_per_img, packing_boundaries):
        packed_num_windows, packed_end_idx = packing_boundaries
        total_windows, window_size, dim = x.shape
        batch_size = len(windows_per_img)

        x = x.reshape(batch_size, -1, dim)
        packed_img_idx = packed_img_idx.reshape(batch_size, -1, PACKED.NUM_METADATA)
        x, _ = self._call_attention(x, attn_mask=attn_mask, packed_img_idx=packed_img_idx)
        x = x.reshape(total_windows, window_size, dim)
        return x

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        packed_img_idx: Optional[torch.Tensor] = None,
        windows_per_img: Optional[torch.Tensor] = None,
        packing_boundaries: Optional[Tuple[List[int], List[int]]] = None,
    ):
        k_x = self.ln_kv(k_x) if hasattr(self, "ln_kv") and k_x is not None else None
        v_x = self.ln_kv(v_x) if hasattr(self, "ln_kv") and v_x is not None else None
        q_x_normalized = self.ln(q_x)

        if self.is_global and packing_boundaries is not None:
            attn_output = self.packed_global_attn(
                q_x_normalized, attn_mask=attn_mask, packed_img_idx=packed_img_idx,
                windows_per_img=windows_per_img, packing_boundaries=packing_boundaries
            )
        else:
            attn_output, weights = self._call_attention(
                q_x_normalized, k_x, v_x, attn_mask=attn_mask,
                packed_img_idx=packed_img_idx,
            )
        x = q_x + self.ls(attn_output)

        return x, q_x_normalized

class MoEKernelWrapper(nn.Module):
    def __init__(
        self,
        in_features: int,
        mlp_width: int,
        shared_module: nn.Module = None,
        num_experts: int = 8,
        k: int = 2,
        split: int = 1,
        use_optimized_swiglu_inference: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts * split - 1
        self.k = k * split - 1
        self._shared_module = shared_module
        self.mlp_width = mlp_width
        self.padded_mlp_width = mlp_width
        self.use_optimized_swiglu_inference = use_optimized_swiglu_inference

        self.w1 = nn.Parameter(torch.empty(self.num_experts, 2 * mlp_width, in_features))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, in_features, mlp_width))
        self._gate_module = nn.Linear(in_features=in_features, out_features=self.num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_experts), persistent=True)

    def pad_weights(self):
        multiple = 16
        pad = (multiple - (self.mlp_width % multiple)) % multiple
        self.padded_mlp_width = self.mlp_width + pad
        w1 = self.w1[:, : self.mlp_width, :]
        w3 = self.w1[:, self.mlp_width :, :]
        w1_padded = torch.nn.functional.pad(w1, (0, 0, 0, pad), "constant", value=0)
        w3_padded = torch.nn.functional.pad(w3, (0, 0, 0, pad), "constant", value=0)
        self.w1 = nn.Parameter(torch.cat([w1_padded, w3_padded], dim=1))
        self.w2 = nn.Parameter(
            torch.nn.functional.pad(self.w2, (0, pad), "constant", value=0)
        )

        if self._shared_module:
            for layer in [self._shared_module.gate_proj, self._shared_module.up_proj]:
                w = layer.weight.data
                layer.weight.data = torch.nn.functional.pad(
                    w, (0, 0, 0, pad), "constant", value=0
                )
                layer.out_features = layer.weight.data.shape[0]
            for layer in [self._shared_module.down_proj]:
                w = layer.weight.data
                layer.weight.data = torch.nn.functional.pad(
                    w, (0, pad), "constant", value=0
                )
                layer.in_features = layer.weight.data.shape[1]

    def forward(self, hidden_states: torch.Tensor):
        bs, sl, d = hidden_states.shape
        hidden_states = hidden_states.view(-1, d)

        router_logits = self._gate_module(hidden_states)
        router_weights = F.sigmoid(router_logits)

        _, topk_ids = torch.topk(router_weights + self.expert_bias.unsqueeze(0), self.k, dim=-1)
        topk_weights = router_weights.gather(-1, topk_ids)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        w1 = self.w1.to(hidden_states.dtype)
        w2 = self.w2.to(hidden_states.dtype)

        if self.use_optimized_swiglu_inference:
            output = routed_ffn_fwd_swiglu_inference(
                x=hidden_states,
                w1=w1[:, : self.padded_mlp_width, :],
                w2=w2,
                w3=w1[:, self.padded_mlp_width : 2 * self.padded_mlp_width, :],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                use_swiglu=True,
            )
        else:
            output = routed_ffn_fwd(
                x=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                use_swiglu=True,
            )

        shared_output = self._shared_module(hidden_states)
        output = (output + shared_output).reshape(bs, sl, d)

        return output

class SingleMLPBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            mlp_width: int,
            bias: bool = True,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, mlp_width, bias=bias)
        self.up_proj = nn.Linear(d_model, mlp_width, bias=bias)
        self.down_proj = nn.Linear(mlp_width, d_model, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable = LayerNorm,
        mixture_of_expert: bool = False,
        num_of_expert: int = 1,
        k: int = 1,
        layer_idx: int = 0,
        moe_layers: Tuple = (),
        split: int = 1,
        gpu_align: bool = False,
    ):
        super().__init__()
        self.mixture_of_expert = mixture_of_expert
        self.num_of_expert = num_of_expert
        self.moe_layers = moe_layers

        self.ln = norm_layer(d_model)

        self.layer_idx = layer_idx
        mlp_width = int(d_model * mlp_ratio)
        mlp_width = math.ceil(mlp_width * 2 // 3 / 16) * 16 if gpu_align else mlp_width * 2 // 3
        if self.mixture_of_expert and layer_idx in moe_layers:
            self.moe = MoEKernelWrapper(
                in_features = d_model,
                mlp_width = math.ceil(mlp_width / split),
                shared_module = SingleMLPBlock(d_model, math.ceil(mlp_width / split), bias=False),
                num_experts = num_of_expert,
                k = k,
                split = split,
            )
        else:
            self.mlp = SingleMLPBlock(d_model, mlp_width)
        self.ls = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        q_x: torch.Tensor,
    ):
        mlp_input = self.ln(x)

        if self.mixture_of_expert and self.layer_idx in self.moe_layers:
            mlp_output = self.moe(mlp_input)
        else:
            mlp_output = self.mlp(mlp_input)
        x = x + self.ls(mlp_output)

        return x

class ResidualAttnWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable = LayerNorm,
        mixture_of_expert: bool = False,
        num_of_expert: int = 1,
        relative_pos_embed_type: str = "",
        is_global: bool = False,
        k: int = 1,
        layer_idx: int = 0,
        moe_layers: Tuple = (),
        split: int = 1,
        gpu_align: bool = False,
    ):
        super().__init__()
        self.attn = AttentionBlock(
            d_model=d_model,
            n_head=n_head,
            norm_layer=norm_layer,
            relative_pos_embed_type=relative_pos_embed_type,
            is_global=is_global,
        )
        self.mlp = MLPBlock(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            mixture_of_expert=mixture_of_expert,
            num_of_expert=num_of_expert,
            k=k,
            layer_idx=layer_idx,
            moe_layers=moe_layers,
            split=split,
            gpu_align=gpu_align,
        )

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        packed_img_idx: Optional[torch.Tensor] = None,
        windows_per_img: Optional[torch.Tensor] = None,
        packing_boundaries: Optional[Tuple[List[int], List[int]]] = None,
    ):
        attn_outputs = self.attn(
            q_x,
            k_x,
            v_x,
            attn_mask,
            packed_img_idx,
            windows_per_img,
            packing_boundaries,
        )

        x, q_x = attn_outputs

        x = self.mlp(x, q_x)

        return x

class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable = LayerNorm,
        mixture_of_expert: bool = False,
        num_of_expert: int = 1,
        global_layers: Sequence[int] = tuple(),
        relative_pos_embed_type: str = "",
        k: int = 1,
        moe_layers: Tuple = (),
        split: int = 1,
        gpu_align: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.relative_pos_embed_type = relative_pos_embed_type
        self.mixture_of_expert = mixture_of_expert
        self.num_of_expert = num_of_expert
        self.global_layers = global_layers
        self.moe_layers = moe_layers

        self.resblocks = nn.ModuleList(
            [
                ResidualAttnWrapper(
                    width,
                    heads,
                    mlp_ratio,
                    norm_layer=norm_layer,
                    mixture_of_expert=self.mixture_of_expert,
                    num_of_expert=self.num_of_expert,
                    relative_pos_embed_type=relative_pos_embed_type,
                    is_global = i in self.global_layers,
                    k=k,
                    layer_idx=i,
                    moe_layers=moe_layers,
                    split=split,
                    gpu_align=gpu_align,
                )
                for i in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        mlp = self.resblocks[0].mlp.expert_modules[0] if self.mixture_of_expert else self.resblocks[0].mlp.mlp
        if hasattr(mlp.up_proj, "int8_original_dtype"):
            return mlp.up_proj.int8_original_dtype
        return mlp.up_proj.weight.dtype

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        packed_img_idx: Optional[torch.Tensor] = None,
        windows_per_img: Optional[torch.Tensor] = None,
        packing_boundaries: Optional[Tuple[List[int], List[int]]] = None,
    ):
        for i, r in enumerate(self.resblocks):
            if packed_img_idx is not None:
                x = torch.where(packed_img_idx[:, :, PACKED.IDX, None] == PACKED.ID_PAD_TOKEN, x[:1, :1, :].detach(), x)

            x = r(
                x,
                attn_mask=attn_mask,
                packed_img_idx=packed_img_idx,
                windows_per_img=windows_per_img,
                packing_boundaries=packing_boundaries,
            )

        if packed_img_idx is not None:
            return x, packed_img_idx
        else:
            return x

class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        no_ln_pre: bool = False,
        final_ln_after_pool: bool = False,
        norm_layer: Callable = partial(LayerNorm, eps=1e-5),
        mixture_of_expert: bool = False,
        num_of_expert: int = 1,
        k: int = 1,
        moe_layers: Tuple = (),
        split: int = 1,
        global_layers: int = None,
        relative_pos_embed_type: str = "",
        use_ln_post: bool = True,
        gpu_align: bool = False,
    ):
        super().__init__()
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = (
            final_ln_after_pool
        )
        self.output_dim = output_dim
        self.heads = heads

        if global_layers and global_layers == -1:
            global_layers = layers
        self.global_layers = global_layers

        self.mixture_of_expert = mixture_of_expert
        self.num_of_expert = num_of_expert

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.relative_pos_embed_type = relative_pos_embed_type

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            norm_layer=norm_layer,
            relative_pos_embed_type=relative_pos_embed_type,
            mixture_of_expert=self.mixture_of_expert,
            num_of_expert=self.num_of_expert,
            global_layers=[] if not self.global_layers else list(range(layers-1, -1, -(layers // self.global_layers)))[:self.global_layers],
            k=k,
            moe_layers=moe_layers,
            split=split,
            gpu_align=gpu_align,
        )

        pool_dim = width

        self.attn_pool = AttentionPooling(
            embed_dim=pool_dim,
            num_heads=attn_pooler_heads,
            norm_layer=norm_layer,
        )

        self.ln_post = norm_layer(pool_dim) if use_ln_post else nn.Identity()
        self.proj = nn.Parameter(torch.randn(pool_dim, output_dim))

    @torch.jit.ignore
    def _packed_pool(self, x: torch.Tensor, packed_img_idx: torch.Tensor, batch_size: int, packing_boundaries: Tuple[List[int], List[int]]):
        packed_num_windows, packed_end_idx = packing_boundaries
        pooled = self.attn_pool(
            x,
            packed_num_windows=packed_num_windows,
            packed_end_idx=packed_end_idx
        ).squeeze(1)
        return pooled

    def image_forward(self, x: torch.Tensor) -> VisionTransformerOutput:
        packed_input = True

        x, packed_img_idx, windows_per_img, packing_boundaries = x
        batch_size = len(windows_per_img)
        total_windows, _, window_size, ph, pw = x.shape

        x = x.reshape(total_windows, -1, window_size*ph, pw)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(total_windows, window_size, -1)

        x = self.ln_pre(x)

        x, packed_img_idx = self.transformer(
            x,
            packed_img_idx=packed_img_idx,
            windows_per_img=windows_per_img,
            packing_boundaries=packing_boundaries
        )

        x = self.ln_post(x)
        latent = tokens = x
        pooled = self._packed_pool(x, packed_img_idx, batch_size, packing_boundaries)

        return pooled, tokens, latent

    def forward(self, x: torch.Tensor) -> VisionTransformerOutput:
        if isinstance(x[0], torch.Tensor):
            pooled, tokens, latent = self.image_forward(x)
            pooled = pooled @ self.proj

            output = VisionTransformerOutput()
            output.pooled = pooled
            return output
        else:
            frames_pooled = []
            for frame_x in x:
                frame_pooled, _, _ = self.image_forward(frame_x)
                frames_pooled.append(frame_pooled)

            pooled = torch.stack(frames_pooled, dim=1).mean(dim=1)

            pooled = pooled @ self.proj
            return pooled


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        norm_layer: Callable = LayerNorm,
        use_ln_post: bool = True,
        gpu_align: bool = False,
    ):
        super().__init__()
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            gpu_align=gpu_align,
        )

        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()

        self.register_buffer("attn_mask", self.build_causal_mask(), persistent=False)

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = self.transformer(x, attn_mask=attn_mask)
        if isinstance(x, Tuple):
            x = x[0]

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text)

        pooled = pooled @ self.text_projection

        return pooled
