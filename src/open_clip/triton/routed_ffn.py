# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from .triton_grouped_gemm import triton_jagged_dense_bmm


def routed_inputs(x, topk_weights, topk_ids, K, E):
    expert, index = topk_ids.contiguous().view(-1).sort(stable=True)

    topk_weights = topk_weights.view(-1)[index]

    index = index // K

    routed_input = x[index]

    zeros = torch.zeros(E, dtype=expert.dtype, device=x.device)
    lengths = zeros.scatter_add(0, expert, torch.ones_like(expert))

    offsets = torch.cumsum(lengths, dim=0, dtype=torch.int64)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=offsets.device), offsets), dim=0
    )

    return routed_input, offsets, index, topk_weights


def routed_ffn_fwd(
    x,
    w1,
    w2,
    topk_weights,
    topk_ids,
    use_swiglu: bool = True,
) -> torch.Tensor:

    B, D = x.shape
    topk = topk_weights.shape[1]

    inter_dim = w2.shape[2]
    E = w1.shape[0]

    if use_swiglu:
        assert w1.shape[1] == 2 * inter_dim, f"{w1.shape=} {w2.shape=} {inter_dim=}"
    else:
        assert w1.shape[1] == inter_dim, f"{w1.shape=} {w2.shape=} {inter_dim=}"

    routed_input, offsets, index, topk_weights = routed_inputs(
        x, topk_weights, topk_ids, topk, E
    )
    act_input = triton_jagged_dense_bmm(
        max_seq_len=B,
        seq_offsets=offsets,
        jagged=routed_input,
        dense=w1.transpose(1, 2),
        bias=None,
    )
    if use_swiglu:
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        act_out = F.silu(gate) * up
    else:
        act_out = F.gelu(act_input)

    grouped_out = triton_jagged_dense_bmm(
        max_seq_len=B,
        seq_offsets=offsets,
        jagged=act_out,
        dense=w2.transpose(1, 2),
        bias=None,
    )
    grouped_out = grouped_out * topk_weights.reshape(-1, 1).to(grouped_out.dtype)

    out = torch.zeros([B, D], dtype=x.dtype, device=x.device)
    out = out.scatter_add_(
        dim=0, index=index.unsqueeze(-1).expand_as(grouped_out), src=grouped_out
    )

    return out
