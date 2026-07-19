from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedTensor:
    packed: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    bits: int
    group_size: int


def _validate(x: torch.Tensor, bits: int, group_size: int) -> None:
    if bits not in (2, 4):
        raise ValueError(f"bits must be 2 or 4, got {bits}")
    if x.ndim < 1 or x.shape[-1] % group_size:
        raise ValueError("last dimension must be divisible by group_size")
    if group_size <= 0 or group_size % (8 // bits):
        raise ValueError("group_size must be positive and pack-aligned")
    if not x.is_floating_point():
        raise TypeError("quantization input must be floating point")


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned codes along the final dimension, low bits first."""
    if bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    per_byte = 8 // bits
    if codes.shape[-1] % per_byte:
        raise ValueError("code dimension is not byte aligned")
    c = codes.to(torch.uint8).reshape(*codes.shape[:-1], -1, per_byte)
    shifts = torch.arange(per_byte, device=c.device, dtype=torch.uint8) * bits
    return torch.sum(c << shifts, dim=-1).to(torch.uint8).contiguous()


def unpack_codes(packed: torch.Tensor, bits: int, elements: int) -> torch.Tensor:
    if bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits
    values = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return values.reshape(*packed.shape[:-1], -1)[..., :elements]


def quantize_tensor(x: torch.Tensor, bits: int, group_size: int = 32) -> QuantizedTensor:
    """Symmetric groupwise quantization suitable for register-level dequantization."""
    _validate(x, bits, group_size)
    shape = tuple(x.shape)
    groups = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    midpoint = 1 << (bits - 1)
    positive_max = midpoint - 1
    scales = groups.abs().amax(dim=-1).clamp_min(1e-8) / positive_max
    codes = torch.round(groups / scales.unsqueeze(-1)).clamp(-midpoint, positive_max)
    codes = (codes + midpoint).to(torch.uint8).reshape(shape)
    return QuantizedTensor(
        packed=pack_codes(codes, bits),
        scales=scales.to(torch.float16).contiguous(),
        original_shape=shape,
        bits=bits,
        group_size=group_size,
    )


def dequantize_tensor(q: QuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    codes = unpack_codes(q.packed, q.bits, q.original_shape[-1]).to(torch.float32)
    midpoint = 1 << (q.bits - 1)
    groups = codes.reshape(*q.original_shape[:-1], -1, q.group_size)
    values = (groups - midpoint) * q.scales.float().unsqueeze(-1)
    return values.reshape(q.original_shape).to(dtype)
