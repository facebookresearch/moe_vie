# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
import triton
import triton.language as tl
from .triton_grouped_gemm import triton_jagged_dense_bmm


def routed_inputs(x, topk_weights, topk_ids, K, E):
    expert, index = topk_ids.contiguous().view(-1).sort(stable=True)

    topk_weights = topk_weights.view(-1)[index]

    index = index // K


    zeros = torch.zeros(E, dtype=expert.dtype, device=x.device)
    lengths = zeros.scatter_add(0, expert, torch.ones_like(expert))

    offsets = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=offsets.device), offsets), dim=0
    )


    return offsets, index, topk_weights


@triton.jit
def silu(A):
    A_SILU = A * tl.sigmoid(A)
    return A_SILU


_configs = []
for BLOCK_M in [64, 128]:
    for BLOCK_N in [64, 128]:
        for BLOCK_K in [32, 64]:
            for num_stages in [2, 3, 5]:
                for num_warps in [4, 8]:
                    _configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": BLOCK_M,
                                "BLOCK_N": BLOCK_N,
                                "BLOCK_K": BLOCK_K,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )


@triton.autotune(
    configs=_configs,
    key=["M", "N", "K"],
)
@triton.jit
def _index_select_jagged_bmm_swiglu(
    seq_offsets,
    Index,
    Jagged,
    Dense,
    Bias,
    Dense_P,
    Bias_P,
    Out,
    M,
    N,
    K,
    A,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_b = tl.program_id(2)
    off_m = tl.program_id(0)
    off_n = tl.program_id(1)

    seq_start = tl.load(seq_offsets + off_b, eviction_policy="evict_last")
    seq_end = tl.load(seq_offsets + off_b + 1, eviction_policy="evict_last")
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Index += seq_start
    Dense += off_b * stride_db
    Dense_P += off_b * stride_db
    Out += seq_start.to(tl.int64) * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    idx_ptrs = Index + offs_m
    idx = tl.load(idx_ptrs, mask=offs_m < seq_len, other=0)

    jg_ptrs = Jagged + idx[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn
    dnp_ptrs = Dense_P + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )
        dnp = tl.load(
            dnp_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )

        accumulator1 += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        accumulator2 += tl.dot(jg, dnp, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk
        dnp_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n
        biasp_ptrs = Bias_P + off_b * stride_bias_b + offs_n

        bias = tl.load(
            bias_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )
        biasp = tl.load(
            biasp_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )

        A = accumulator1 + bias[None, :].to(tl.float32)
        B = accumulator2 + biasp[None, :].to(tl.float32)
    else:
        A = accumulator1
        B = accumulator2

    A_SILU = silu(A)

    out = (A_SILU * B).to(Out.dtype.element_ty)

    out_ptrs = Out + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]

    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


def index_select_jagged_bmm_swiglu(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_p: torch.Tensor,
    bias_p: Optional[torch.Tensor],
) -> torch.Tensor:
    L, A = index.shape
    _, K = jagged.shape
    E, _, N = weight.shape
    output = torch.empty(
        (L * A, N), dtype=jagged.dtype, device=jagged.device
    )

    grid = lambda meta: (
        triton.cdiv(max_seq_len, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        E,
    )

    _index_select_jagged_bmm_swiglu[grid](
        seq_offsets=offsets,
        Index=index,
        Jagged=jagged,
        Dense=weight,
        Bias=bias,
        Dense_P=weight_p,
        Bias_P=bias_p,
        Out=output,
        M=max_seq_len,
        N=N,
        K=K,
        A=A,
        stride_jm=jagged.stride(0),
        stride_db=weight.stride(0),
        stride_dk=weight.stride(1),
        stride_dn=weight.stride(2),
        stride_bias_b=bias.stride(0) if bias is not None else 0,
        stride_om=output.stride(0),
        HAS_BIAS=bias is not None,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    return output


def routed_ffn_fwd(
    x,
    w1,
    w3,
    w2,
    topk_weights,
    topk_ids,
    use_swiglu: bool = True,
) -> torch.Tensor:

    B, D = x.shape
    topk = topk_weights.shape[1]

    E = w1.shape[0]

    offsets, index, topk_weights = routed_inputs(x, topk_weights, topk_ids, topk, E)

    act_out = index_select_jagged_bmm_swiglu(
        max_seq_len=B,
        offsets=offsets,
        index=index.view(-1, topk),
        jagged=x,
        weight=w1.transpose(1, 2),
        bias=None,
        weight_p=w3.transpose(1, 2),
        bias_p=None,
    )

    grouped_out = triton_jagged_dense_bmm(
        max_seq_len=B,
        seq_offsets=offsets,
        jagged=act_out,
        dense=w2.transpose(1, 2),
        bias=None,
        topk_weights=topk_weights.reshape(-1, 1).to(x.dtype),
    )

    out = torch.zeros([B, D], dtype=x.dtype, device=x.device)
    out = out.scatter_add_(
        dim=0, index=index.unsqueeze(-1).expand_as(grouped_out), src=grouped_out
    )

    return out
