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
from quark.torch.quantization.config.config import RocFP4Spec
from quark.torch.quantization.tensor_quantize import DynamicScaledFakeQuantize

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["quant_dequant_rocfp4"]

# Global quantizer instance (created on first use)
_rocfp4_quantizer = None


def quant_dequant_rocfp4(
    x: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """
    Simulate RocFP4 quantization by quantizing and immediately dequantizing.
    """
    global _rocfp4_quantizer

    if _rocfp4_quantizer is None:
        rocfp4_spec = RocFP4Spec(ch_axis=-1, group_size=group_size, is_dynamic=True)
        rocfp4_qtensor_config = rocfp4_spec.to_quantization_spec()
        _rocfp4_quantizer = DynamicScaledFakeQuantize(
            rocfp4_qtensor_config, device=torch.get_default_device()
        )

    # Apply quantize-dequantize
    return _rocfp4_quantizer(x)
