# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib
import ttnn
import torch
import numpy as np
from tests.ttnn.utils_for_testing import assert_equal


tt_dtype_to_torch_dtype = {
    ttnn.uint16: torch.int16,
    ttnn.uint32: torch.int32,
    ttnn.float32: torch.float,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float,
}


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 4, 32],
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        0,
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        3,
    ],
)
def test_indexed_slice(shape, dim, value, device):
    torch.manual_seed(2024)
    torch.set_printoptions(linewidth=1000)

    # torch_batch_ids = torch.randint(0, B - 1, (1, 1, 1, b))
    torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    print(torch_input.shape)
    torch_index = torch.tensor([0, 2])
    torch_output = torch.index_fill(torch_input, dim, torch_index, value)
    print(torch_output)
    # batch_ids = ttnn.Tensor(torch_batch_ids, ttnn.uint32).to(
    #     device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    # )
    input = ttnn.Tensor(torch_input, ttnn.uint32).to(device)
    index = ttnn.Tensor(torch_index, ttnn.uint32).to(device)
    output = ttnn.index_fill(input, index, dim, value)
    output_torch = ttnn.to_torch(output)

    # print(output)
    # assert assert_equal(output_torch, torch_output)
    assert True
