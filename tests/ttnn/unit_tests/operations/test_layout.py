# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_layout(device, batch_size, model_location_generator, reset_seeds):
    num_classes = 10

    x = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
    x = x.permute(0, 2, 3, 1)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)

    weight = torch.randn((6, 1, 5, 5), dtype=torch.bfloat16)
    bias = torch.randn((1, 1, 1, 6), dtype=torch.bfloat16)

    weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

    y = device_out(x, batch_size, num_classes, device, weight, bias, reset_seeds)

    z = device_in(x, batch_size, num_classes, device, weight, bias, reset_seeds)

    z = ttnn.to_torch(z)
    y = ttnn.to_torch(y)

    assert_with_pcc(y, z, 0.99)


def conv(device, input_tensor, batch_size, weight, bias):
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )
    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight,
        in_channels=input_tensor.shape[3],
        out_channels=weight.shape[0],
        device=device,
        bias_tensor=bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=True,
        groups=1,
    )
    return x, out_height, out_width


def device_out(input_tensor, batch_size, num_classes, device, weight, bias, reset_seeds):
    conv1, out_height, out_width = conv(device, input_tensor, batch_size, weight, bias)

    conv1 = ttnn.from_device(conv1)
    conv1 = ttnn.to_layout(conv1, layout=ttnn.ROW_MAJOR_LAYOUT)

    return conv1


def device_in(input_tensor, batch_size, num_classes, device, weight, bias, reset_seeds):
    conv2, out_height, out_width = conv(device, input_tensor, batch_size, weight, bias)

    conv2 = ttnn.to_layout(conv2, layout=ttnn.ROW_MAJOR_LAYOUT)

    return conv2
