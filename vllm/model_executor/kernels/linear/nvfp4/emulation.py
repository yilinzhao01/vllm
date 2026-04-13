# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    kE2M1ToFloat_handle,
    run_nvfp4_emulations,
)
from vllm.logger import init_logger
from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
logger = init_logger(__name__)

class EmulationNvFp4LinearKernel(NvFp4LinearKernel):
    """Software emulation fallback for NVFP4 (dequant → BF16 matmul)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Always available as a last-resort fallback.
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Move the E2M1 lookup table to the device now, because
        # `.to(device)` is not allowed during CUDA graph capture.
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)

        if layer.emulation_dequantize_weights:
            logger.warning_once("Dequantizing NVFP4 linear weights ahead of time with emulation_dequantize_weights=True.")
            
            dq_w = dequantize_to_dtype(
                tensor_fp4=layer.weight,
                tensor_sf=layer.weight_scale,
                global_scale=layer.weight_global_scale,
                dtype=torch.get_default_dtype(),
                block_size=16,
                swizzle=False,
            )
            layer.weight = nn.Parameter(dq_w, requires_grad=False)
            layer.weight_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = run_nvfp4_emulations(
            x=x,
            input_global_scale=layer.input_global_scale_inv,
            weight=layer.weight,
            weight_scale_swizzled=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            swizzle=False,
        )
        if bias is not None:
            out = out + bias
        return out
