import pytest
import torch

from kvssd.quant import dequantize_tensor, quantize_tensor


@pytest.mark.parametrize(("bits", "max_mae"), [(4, 0.12), (2, 0.65)])
def test_quantization_round_trip(bits: int, max_mae: float) -> None:
    x = torch.randn(3, 2, 128, generator=torch.Generator().manual_seed(1))
    quantized = quantize_tensor(x, bits, group_size=32)
    restored = dequantize_tensor(quantized)
    assert restored.shape == x.shape
    assert torch.mean(torch.abs(x - restored)).item() < max_mae
    assert quantized.packed.numel() == x.numel() * bits // 8


def test_rejects_invalid_layout() -> None:
    with pytest.raises(ValueError, match="divisible"):
        quantize_tensor(torch.randn(2, 31), 4, group_size=16)
