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
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)

__all__ = ["QuarkROCFP4"]

logger = init_logger(__name__)


class QuarkROCFP4(QuarkScheme):
    """
    Quark ROCFP4 quantization scheme.

    Supports loading ROCFP4 checkpoints with the following structure:
    - weight: uint8, shape [out_features, in_features // 2] (packed FP4)
    - weight_scale: uint8, shape [out_features, in_features // group_size] (FP8 E5M3)
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

        # Per-group weight scale (FP8 E5M3 stored as uint8)
        # Shape: [out_features, in_features // group_size]
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,  # FP8 E5M3 stored as uint8
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

        # TODO: use a single functional call to unpack and dequantize weights,
        # currently ugly.
        weight_scale_dq = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
            "fp8_e5m3",
            layer.weight_scale.data,
            None,  # scale
            None,  # zero_point
            -1,  # ch_axis
            1,  # group_size
            "per_group",  # qscheme
        )

        packing_instance = Pack_fp4(None, "fp4")

        weight = packing_instance.unpack(
            layer.weight.data,
            reorder=None,
        )

        weight_dq = quark.torch.kernel.dequantize(
            "fp4", weight, weight_scale_dq, None, -1, 16, "per_group"
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
