# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import quark.torch.kernel
import torch
from quark.torch.quantization.config.config import RocFP4Spec
from quark.torch.quantization.tensor_quantize import DynamicScaledFakeQuantize
from quark.torch.utils.pack import Pack_fp4
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.quark.schemes.quark_scheme import (
    QuarkScheme,
)
from vllm.model_executor.layers.quantization.utils.rocfp4_utils import (
    quant_dequant_rocfp4_global,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)

__all__ = ["QuarkROCFP4", "QuarkROCFP4Global"]

logger = init_logger(__name__)


class QuarkROCFP4(QuarkScheme):
    """
    Quark ROCFP4 quantization scheme.

    Supports loading ROCFP4 checkpoints with the following structure:
    - weight: uint8, shape [out_features, in_features // 2] (packed FP4)
    - weight_scale: shape [out_features, in_features // group_size], FP8 E5M3
      stored as uint8, bf16, or fp32
    """

    def __init__(self):
        self.group_size = 16

        # TODO: Avoid using quark DynamicScaledFakeQuantize,
        # use a functional entrypoint for rocfp4 QDQ.
        rocfp4_spec = RocFP4Spec(ch_axis=-1, group_size=16, is_dynamic=True)
        rocfp4_qtensor_config = rocfp4_spec.to_quantization_spec()
        self.rocfp4_quantizer = DynamicScaledFakeQuantize(
            rocfp4_qtensor_config, device=torch.get_default_device()
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must be "
                f"divisible by group size ({self.group_size})"
            )

        # Weight: FP4 packed as uint8 (2 FP4 values per uint8)
        # Shape: [out_features, in_features // 2]
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,  # 2 FP4 values per uint8
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Per-group weight scale: FP8 E5M3 stored as uint8, bf16, or fp32
        # (fp32 buffer holds all three).
        # Shape: [out_features, in_features // group_size]
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

        weight_scale_dq = layer.weight_scale.data
        # FP8 E5M3 scales load into the fp32 buffer as integer byte codes;
        # bf16/fp32 ckpts already hold the dequantized scale.
        if (weight_scale_dq == weight_scale_dq.round()).all():
            weight_scale_dq = quark.torch.kernel.dequantize(
                "fp8_e5m3",
                weight_scale_dq.to(torch.uint8),
                None,
                None,
                -1,
                1,
                "per_group",
            )

        packing_instance = Pack_fp4(None, "fp4")
        weight = packing_instance.unpack(layer.weight.data, reorder=None)
        weight_dq = quark.torch.kernel.dequantize(
            "fp4", weight, weight_scale_dq, None, -1, self.group_size, "per_group"
        )
        weight_dq = weight_dq.to(layer.params_dtype)

        layer.weight = Parameter(weight_dq, requires_grad=False)
        layer.weight_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO: by default, dequantize weights in the forward.
        x = self.rocfp4_quantizer(x)

        output = torch.nn.functional.linear(x, layer.weight, bias)

        return output


class QuarkROCFP4Global(QuarkROCFP4):
    """
    Quark rocfp4_global quantization scheme: effective per-group scale =
    e5m3_dequant(weight_scale) * weight_scale_2 (per-tensor global).
    """

    def __init__(self, group_size: int = 16, input_global_static: bool = False):
        super().__init__()
        self.group_size = group_size
        self.input_global_static = input_global_static

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        super().create_weights(
            layer,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            **kwargs,
        )
        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        if self.input_global_static:
            input_scale_2 = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale_2", input_scale_2)

    def _weight_global_scale(self, layer: torch.nn.Module) -> torch.Tensor:
        # One global scale per output partition; expand to per-row [out_features, 1].
        widths = torch.tensor(
            layer.logical_widths, device=layer.weight_scale_2.device
        )
        per_row = layer.weight_scale_2.data.repeat_interleave(widths)
        return per_row.unsqueeze(-1)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.input_global_static:
            layer.input_global_scale = Parameter(
                layer.input_scale_2.max().to(torch.float32), requires_grad=False
            )
            del layer.input_scale_2
        # Dequantize the E5M3 byte codes, apply the global, then defer to the base.
        scale = quark.torch.kernel.dequantize(
            "fp8_e5m3", layer.weight_scale.data.to(torch.uint8),
            None, None, -1, 1, "per_group",
        )
        layer.weight_scale.data = scale * self._weight_global_scale(layer)
        super().process_weights_after_loading(layer)
        layer.weight_scale_2 = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global_scale = layer.input_global_scale if self.input_global_static else None
        x = quant_dequant_rocfp4_global(x, self.group_size, global_scale=global_scale)
        return torch.nn.functional.linear(x, layer.weight, bias)
