# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from loguru import logger


def get_tensor(
    shape,
    device,
    *,
    npu_layout=ttnn.TILE_LAYOUT,
    npu_dtype=ttnn.bfloat16,
    zero=False,
    need_padding=True,
    tensor_in_dram=True,
):
    cpu_dtype = torch.float

    torch_tensor = torch.rand(shape, dtype=cpu_dtype) + 1 if not zero else torch.zeros(shape, dtype=cpu_dtype)
    torch_tensor.requires_grad_()
    tt_tensor = (
        ttnn.Tensor(torch_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        if need_padding
        else ttnn.Tensor(torch_tensor, npu_dtype).to(npu_layout).to(device)
    )
    return torch_tensor, tt_tensor


def to_cpu(npu_tensor, shape, *, cpu_layout=ttnn.ROW_MAJOR_LAYOUT):
    if npu_tensor is None:
        return None
    if not isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(shape).to_torch()
    return cpu_tensor


@pytest.mark.parametrize(
    "shape",
    (
        [32, 32],
        [32, 64],
        [31, 5120],
    ),
    ids=["32, 32", "32, 64", "31, 5120"],
)
def test_uint8_and_bfloat16(shape, device):
    passing = True
    # prepare tensors
    torch_input, tt_input = get_tensor(shape, device)
    _, tt_output = get_tensor(shape, device)
    torch_uint8_tensor = torch.full(torch_input.shape, 1, dtype=torch.uint8)
    tt_uint8_tensor = (
        ttnn.Tensor(torch_uint8_tensor, ttnn.uint8).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)
    )

    # run test op
    ttnn.experimental.operations.primary.moreh_test2(tt_input, tt_uint8_tensor, output=tt_output)
    tt_input_cpu = to_cpu(tt_input, shape)
    tt_output_cpu = to_cpu(tt_output, shape)

    # comparison
    rtol = 0.02
    atol = 0.2
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(tt_input_cpu, tt_output_cpu, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"output passing={passing}")
    logger.debug(f"output pcc={output_pcc}")
    logger.debug(f"diff min: {(tt_input_cpu - tt_output_cpu).min()}, max: {(tt_input_cpu - tt_output_cpu).max()}")

    torch.set_printoptions(threshold=1000000, linewidth=100000000, sci_mode=False, precision=2)
    # print out the second tile when shape is [32, 64]
    if passing == False and shape[1] == 64:
        split_tensors = torch.split(tt_output_cpu, 32, dim=1)
        for i, split_tensor in enumerate(split_tensors):
            logger.debug(f"Tensor {i+1}:")
            logger.debug(split_tensor)

    assert passing
