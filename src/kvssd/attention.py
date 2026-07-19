from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .backends import AttentionBackend, PackedBatch, resolve_attention_backend
from .format import PackedKVBlock
from .quant import dequantize_tensor


def reference_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Decode attention with query [QH,D] and K/V [T,KVH,D], including GQA."""
    if query.ndim != 2 or keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError("expected query [QH,D] and matching keys/values [T,KVH,D]")
    q_heads, dim = query.shape
    if keys.shape[2] != dim or q_heads % keys.shape[1]:
        raise ValueError("head_dim mismatch or query heads not divisible by KV heads")
    mapping = torch.arange(q_heads, device=query.device) // (q_heads // keys.shape[1])
    expanded_k = keys[:, mapping, :]
    expanded_v = values[:, mapping, :]
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(dim)
    scores = torch.einsum("hd,thd->ht", query.float(), expanded_k.float()) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", weights, expanded_v.float()).to(query.dtype)


def packed_attention_reference(
    query: torch.Tensor,
    blocks: Sequence[PackedKVBlock],
    sm_scale: float | None = None,
) -> torch.Tensor:
    if not blocks:
        raise ValueError("at least one block is required")
    keys = torch.cat(
        [dequantize_tensor(x.key, torch.float32)[: x.valid_tokens] for x in blocks], dim=0
    ).to(query.device)
    values = torch.cat(
        [dequantize_tensor(x.value, torch.float32)[: x.valid_tokens] for x in blocks], dim=0
    ).to(query.device)
    return reference_attention(query, keys, values, sm_scale)


def stack_blocks(blocks: Sequence[PackedKVBlock], device: torch.device) -> tuple[torch.Tensor, ...]:
    if not blocks:
        raise ValueError("at least one block is required")
    kwargs = {"device": device, "non_blocking": device.type == "cuda"}
    return (
        torch.stack([x.key.packed for x in blocks]).to(**kwargs),
        torch.stack([x.value.packed for x in blocks]).to(**kwargs),
        torch.stack([x.key.scales for x in blocks]).to(**kwargs),
        torch.stack([x.value.scales for x in blocks]).to(**kwargs),
        torch.tensor([x.valid_tokens for x in blocks], dtype=torch.int32, device=device),
    )


def cuda_extension_available() -> bool:
    try:
        resolve_attention_backend("cuda", torch.device("cuda"))
        return True
    except RuntimeError:
        return False


def fused_attention_tensors(
    query: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    scales_k: torch.Tensor,
    scales_v: torch.Tensor,
    valid_tokens: torch.Tensor,
    bits: int,
    group_size: int,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-level CUDA entry point returning normalized output and log-sum-exp statistics."""
    backend = resolve_attention_backend("cuda", query.device)
    return backend.run(
        query,
        PackedBatch(packed_k, packed_v, scales_k, scales_v, valid_tokens, bits, group_size),
        sm_scale,
    )


def merge_attention_chunks(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly merge independently normalized attention chunks using log normalizers."""
    merged_lse = torch.logaddexp(left_lse, right_lse)
    merged_output = left_output * torch.exp(left_lse - merged_lse).unsqueeze(
        -1
    ) + right_output * torch.exp(right_lse - merged_lse).unsqueeze(-1)
    return merged_output, merged_lse


def packed_attention(
    query: torch.Tensor,
    blocks: Sequence[PackedKVBlock],
    sm_scale: float | None = None,
    require_cuda_kernel: bool = False,
    attention_backend: str | AttentionBackend | None = None,
) -> torch.Tensor:
    """Run packed attention through an explicit or automatically selected backend."""
    if not blocks:
        raise ValueError("at least one block is required")
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if require_cuda_kernel and attention_backend not in (None, "auto", "cuda"):
        raise ValueError("require_cuda_kernel conflicts with a non-CUDA attention backend")
    requested = "cuda" if require_cuda_kernel else attention_backend
    backend = resolve_attention_backend(requested, query.device)
    kp, vp, ks, vs, valid = stack_blocks(blocks, query.device)
    output, _lse = backend.run(
        query,
        PackedBatch(
            kp,
            vp,
            ks,
            vs,
            valid,
            blocks[0].key.bits,
            blocks[0].key.group_size,
        ),
        scale,
    )
    return output.to(query.dtype)
