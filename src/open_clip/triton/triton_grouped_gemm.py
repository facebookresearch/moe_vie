# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import types
import torch
import triton
import triton.language as tl
from torch.utils.flop_counter import register_flop_formula


def custom_triton_op(qualname, mutates_args):
    def wrapper(func):
        try:
            op_exists = torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "CUDA")
        except Exception:
            op_exists = False

        if op_exists is False:
            return torch._library.triton_op(qualname, func, mutates_args=mutates_args)
        else:
            return func

    return wrapper


def custom_register_kernel(qualname, device_types):
    def wrapper(func):
        try:
            op_exists = torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "CPU")
        except Exception:
            op_exists = False

        if op_exists is False:
            return torch.library.register_kernel(qualname, device_types, func)
        else:
            return func

    return wrapper


@custom_register_kernel("moe::triton_grouped_gemm", "cpu")
def cpu_triton_jagged_dense_bmm(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if jagged.stride(-1) != 1:
        jagged = jagged.contiguous()
    L, D = jagged.shape
    B, _, K = dense.shape
    bmm_out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
    for i in range(B):
        begin = seq_offsets[i]
        end = seq_offsets[i + 1]
        bmm_out[begin:end, :] = torch.mm(jagged[begin:end, :], dense[i, :, :])
        if bias is not None:
            bmm_out[begin:end, :] += bias[i, :]

    return bmm_out


@custom_register_kernel("moe::triton_grouped_gemm_backward", "cpu")
def cpu_jagged_dense_bmm_dense_backward(
    d_bmm_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    max_seq_len: int,
    bias: torch.Tensor | None,
    has_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_jagged = torch.empty_like(jagged)
    d_dense = torch.empty_like(dense)
    B = dense.size(0)

    for i in range(B):
        begin = seq_offsets[i]
        end = seq_offsets[i + 1]
        d_jagged[begin:end, :] = torch.mm(d_bmm_out[begin:end, :], dense[i, :, :].T)
        d_dense[i, :, :] = torch.mm(jagged[begin:end, :].T, d_bmm_out[begin:end, :])

    d_bias = torch.empty_like(bias) if bias is not None else None

    return d_jagged, d_dense, d_bias, None


CONFIGS_JAGGED_DENSE: list[triton.runtime.autotuner.Config] = (
    [
        triton.Config(
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in [64, 128]
        for BLOCK_N in [64, 128, 256]
        for BLOCK_K in [32, 64]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ]
    if os.environ.get("MOE_DISABLE_AUTOTUNE", "0") == "0"
    else [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        )
    ]
)


@triton.autotune(
    CONFIGS_JAGGED_DENSE,
    key=["M", "N", "K"],
)
@triton.jit
def jagged_dense_bmm_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Residual,
    Topk_Weights,
    Out,
    M,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_rm,
    stride_om,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_TOPK_WEIGHTS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(1)
    pid_out_b = pid_b
    pid_mn = tl.program_id(0)
    seq_start = tl.load(seq_offsets + pid_b)
    seq_end = tl.load(seq_offsets + pid_b + 1)
    seq_len = seq_end - seq_start

    input_start = seq_start
    output_start = seq_start

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += input_start * stride_jm
    Dense += pid_out_b * stride_db
    Out += output_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            mask=(offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )

        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    out = accumulator.to(Out.dtype.element_ty)
    if HAS_BIAS:
        bias_ptrs = Bias + pid_out_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        out += bias[None, :]

    if HAS_RESIDUAL:
        residual_ptrs = (
            Residual + output_start * stride_rm + offs_m[:, None] * stride_rm + offs_n
        )
        residual = tl.load(
            residual_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_n < N)
        )
        out += residual
    if HAS_TOPK_WEIGHTS:
        topk_weights_ptrs = Topk_Weights + output_start + offs_m
        topk_weights = tl.load(topk_weights_ptrs, mask=(offs_m < seq_len))
        out *= topk_weights[:, None]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N))


CONFIGS_JAGGED_JAGGED: list[triton.runtime.autotuner.Config] = (
    [
        triton.Config(
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in [64, 128]
        for BLOCK_N in [64, 128, 256]
        for BLOCK_K in [32, 64]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ]
    if os.environ.get("MOE_DISABLE_AUTOTUNE", "0") == "0"
    else [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        )
    ]
)


@triton.autotune(
    CONFIGS_JAGGED_JAGGED,
    key=["M", "N", "K"],
)
@triton.jit
def _jagged_jagged_bmm_reduce_sum(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    ReduceOut,
    M,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    REDUCE_JAGGEDB: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_b = tl.program_id(1)
    seq_start = tl.load(seq_offsets + pid_b)
    seq_end = tl.load(seq_offsets + pid_b + 1)
    seq_len = seq_end - seq_start

    a_start = seq_start

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    Out += pid_b * stride_ob
    JaggedA += a_start * stride_ak
    JaggedB += seq_start * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_safe = offs_m % M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_safe = offs_n % N

    if REDUCE_JAGGEDB:
        out_reduce_ptrs = ReduceOut + pid_b * stride_orb + offs_n_safe * stride_orn
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + offs_m_safe[:, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n_safe[None, :]

    for k in range(seq_len, 0, -BLOCK_K):
        jg_a = tl.load(
            jg_a_ptrs,
            mask=(offs_k[None, :] < k),
            other=0.0,
        )

        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_k[:, None] < k),
            other=0.0,
        )

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if pid_m == 0:
                acc_reduce += tl.sum(jg_b.to(tl.float32), axis=0)

        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if REDUCE_JAGGEDB:
        if pid_m == 0:
            tl.store(
                out_reduce_ptrs,
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )


@custom_triton_op("moe::triton_grouped_gemm", mutates_args=())
def triton_jagged_dense_bmm(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if jagged.stride(-1) != 1:
        jagged = jagged.contiguous()
    if residual is not None and residual.stride(-1) != 1:
        residual = residual.contiguous()
    L, D = jagged.shape
    B, _, K = dense.shape

    bmm_out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)

    grid = lambda meta: (
        triton.cdiv(max_seq_len, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_N"]),
        B,
    )

    torch._library.capture_triton(jagged_dense_bmm_add_kernel)[grid](
        seq_offsets=seq_offsets,
        Jagged=jagged,
        Dense=dense,
        Bias=bias,
        Residual=residual,
        Topk_Weights=topk_weights,
        Out=bmm_out,
        M=max_seq_len,
        N=K,
        K=D,
        stride_jm=jagged.stride(0),
        stride_db=dense.stride(0),
        stride_dk=dense.stride(1),
        stride_dn=dense.stride(2),
        stride_bias_b=bias.stride(0) if bias is not None else 0,
        stride_rm=residual.stride(0) if residual is not None else 0,
        stride_om=bmm_out.stride(0),
        HAS_BIAS=bias is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_TOPK_WEIGHTS=topk_weights is not None,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    return bmm_out


@custom_triton_op("moe::triton_grouped_gemm_backward", mutates_args=())
def jagged_dense_bmm_dense_backward(
    d_bmm_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    max_seq_len: int,
    bias: torch.Tensor | None,
    has_residual: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, _, K = dense.shape
    L, D = jagged.shape

    d_jagged = torch.empty_like(jagged)
    d_dense = torch.empty_like(dense)
    d_bias = torch.empty_like(bias) if bias is not None else None
    d_residual = d_bmm_out.clone() if has_residual else None

    grid = lambda meta: (
        triton.cdiv(max_seq_len, meta["BLOCK_M"]) * triton.cdiv(D, meta["BLOCK_N"]),
        B,
    )

    torch._library.capture_triton(jagged_dense_bmm_add_kernel)[grid](
        seq_offsets=seq_offsets,
        Jagged=d_bmm_out,
        Dense=dense,
        Bias=None,
        Residual=None,
        Topk_Weights=None,
        Out=d_jagged,
        M=max_seq_len,
        N=D,
        K=K,
        stride_jm=d_bmm_out.stride(0),
        stride_db=dense.stride(0),
        stride_dk=dense.stride(2),
        stride_dn=dense.stride(1),
        stride_bias_b=0,
        stride_rm=0,
        stride_om=d_jagged.stride(0),
        HAS_BIAS=False,
        HAS_RESIDUAL=False,
        HAS_TOPK_WEIGHTS=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    grid = lambda meta: (
        triton.cdiv(D, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_N"]),
        B,
    )

    torch._library.capture_triton(_jagged_jagged_bmm_reduce_sum)[grid](
        seq_offsets=seq_offsets,
        JaggedA=jagged,
        JaggedB=d_bmm_out,
        Out=d_dense,
        ReduceOut=d_bias,
        M=D,
        N=K,
        K=max_seq_len,
        stride_ak=jagged.stride(0),
        stride_bk=d_bmm_out.stride(0),
        stride_ob=d_dense.stride(0),
        stride_om=d_dense.stride(1),
        stride_on=d_dense.stride(2),
        stride_orb=d_bias.stride(0) if d_bias is not None else 0,
        stride_orn=d_bias.stride(1) if d_bias is not None else 0,
        REDUCE_JAGGEDB=bias is not None,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    return (
        d_jagged,
        d_dense,
        d_bias,
        d_residual,
    )


def _jagged_dense_bmm_dense_backward(
    ctx, d_bmm_out: torch.Tensor
) -> tuple[
    None,
    None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        seq_offsets,
        jagged,
        dense,
        bias,
    ) = ctx.saved_tensors
    (
        d_jagged,
        d_dense,
        d_bias,
        d_residual,
    ) = jagged_dense_bmm_dense_backward(
        d_bmm_out,
        seq_offsets,
        jagged,
        dense,
        ctx.max_seq_len,
        bias,
        ctx.has_residual,
    )
    return (
        None,
        None,
        d_jagged,
        d_dense,
        d_bias,
        d_residual,
    )


def _jagged_dense_bmm_setup_context(ctx, inputs, output):
    (
        max_seq_len,
        seq_offsets,
        jagged,
        dense,
        bias,
        residual,
        topk_weights,
    ) = inputs
    L, D = jagged.shape
    B, _, K = dense.shape

    ctx.save_for_backward(
        seq_offsets,
        jagged,
        dense,
        bias,
    )
    ctx.max_seq_len = max_seq_len
    ctx.has_residual = residual is not None


if not isinstance(
    triton_jagged_dense_bmm, types.FunctionType
):
    triton_jagged_dense_bmm.register_autograd(
        _jagged_dense_bmm_dense_backward, setup_context=_jagged_dense_bmm_setup_context
    )


def _jagged_dense_bmm_flops(dense_shape, jagged_len):
    b, k, n = dense_shape
    return k * n * jagged_len * 2


@register_flop_formula(torch.ops.moe.triton_grouped_gemm, get_raw=True)
def jagged_dense_bmm_flops(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    *args,
    **kwargs,
):
    return _jagged_dense_bmm_flops(dense.shape, jagged.size(0))


@register_flop_formula(torch.ops.moe.triton_grouped_gemm_backward, get_raw=True)
def jagged_dense_bmm_bwd_flops(
    d_bmm_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    max_seq_len: int,
    *args,
    **kwargs,
):
    return _jagged_dense_bmm_flops(dense.shape, jagged.size(0)) * 2
