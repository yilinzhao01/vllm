# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for RocFP4 quantization.

RocFP4 uses:
- FP4 data format (4-bit floating point)
- FP8 E5M3 scales (8-bit floating point for scales)
- Per-group quantization with group_size=16
"""

import torch

try:
    from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
        fake_quantize_fp4_fp6_per_group_with_scale,
        fake_quantize_fp8_e5m3_per_tensor_with_scale,
    )
    from quark.torch.quantization.config.config import RocFP4Spec
    from quark.torch.quantization.tensor_quantize import DynamicScaledFakeQuantize
except ImportError as _quark_import_err:
    fake_quantize_fp4_fp6_per_group_with_scale = None
    fake_quantize_fp8_e5m3_per_tensor_with_scale = None
    RocFP4Spec = None
    DynamicScaledFakeQuantize = None
    _QUARK_IMPORT_ERR = _quark_import_err
else:
    _QUARK_IMPORT_ERR = None

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "quant_dequant_rocfp4",
    "quant_dequant_rocfp4_global",
]

# Global quantizer instance (created on first use)
_rocfp4_quantizer = None

_FP4_QUANT_MAX = 6.0
_E5M3_QUANT_MAX = 114688.0
_SCALE_EPS = torch.finfo(torch.float32).eps


def quant_dequant_rocfp4(
    x: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """
    Simulate RocFP4 quantization by quantizing and immediately dequantizing.
    """
    global _rocfp4_quantizer

    if _QUARK_IMPORT_ERR is not None:
        raise RuntimeError(
            "RocFP4 quantization requires `quark` with RocFP4Spec / "
            "DynamicScaledFakeQuantize, which is not available in the "
            "installed `amd-quark`."
        ) from _QUARK_IMPORT_ERR

    if _rocfp4_quantizer is None:
        rocfp4_spec = RocFP4Spec(ch_axis=-1, group_size=group_size, is_dynamic=True)
        rocfp4_qtensor_config = rocfp4_spec.to_quantization_spec()
        _rocfp4_quantizer = DynamicScaledFakeQuantize(
            rocfp4_qtensor_config, device=torch.get_default_device()
        )

    # Apply quantize-dequantize
    return _rocfp4_quantizer(x)


def quant_dequant_rocfp4_global(
    x: torch.Tensor,
    group_size: int = 16,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Two-stage rocfp4_global activation QDQ: FP4 per-group, then the per-group
    scale is requantized to FP8 E5M3 through a global scale.
    """
    if _QUARK_IMPORT_ERR is not None:
        raise RuntimeError(
            "RocFP4 global quantization requires `quark` with the "
            "hw_emulation fake_quantize entrypoints, which are not available "
            "in the installed `amd-quark`."
        ) from _QUARK_IMPORT_ERR

    xf = x.detach().to(torch.float32)
    block_scale = (
        xf.reshape(*xf.shape[:-1], xf.shape[-1] // group_size, group_size)
        .abs()
        .amax(-1)
        / _FP4_QUANT_MAX
    )
    block_scale = block_scale.masked_fill(block_scale == 0.0, _SCALE_EPS)
    if global_scale is None:
        global_scale = block_scale.amax() / _E5M3_QUANT_MAX
    else:
        global_scale = global_scale.to(torch.float32)
    eff_scale = fake_quantize_fp8_e5m3_per_tensor_with_scale(block_scale, global_scale)
    return fake_quantize_fp4_fp6_per_group_with_scale(
        x, eff_scale, -1, group_size, "fp4"
    )
