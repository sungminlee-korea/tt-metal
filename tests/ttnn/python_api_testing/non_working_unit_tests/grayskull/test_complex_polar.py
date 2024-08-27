# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from itertools import product
from functools import partial

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_complex_polar_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.complex(
        torch.Tensor(size=input_shape[0]).uniform_(-100, 100), torch.Tensor(size=input_shape[0]).uniform_(-100, 100)
    )

    if dtype[0] == ttnn.bfloat8_b:
        x.real = ttnn.to_torch(
            ttnn.from_torch(x.real, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None)
        ).to(torch.float)

        x.imag = ttnn.to_torch(
            ttnn.from_torch(x.imag, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None)
        ).to(torch.float)
    else:
        x.real = x.real.to(torch.bfloat16).to(torch.float)
        x.imag = x.imag.to(torch.bfloat16).to(torch.float)

    try:
        # get ref result
        ref_value = torch.polar(x.real, x.imag)

        x = ttnn.complex_tensor(
            ttnn_ops.setup_ttnn_tensor(x.real, device, dlayout[0], in_mem_config[0], dtype[0]),
            ttnn_ops.setup_ttnn_tensor(x.imag, device, dlayout[0], in_mem_config[0], dtype[0]),
        )

        tt_result = ttnn.polar(x, memory_config=output_mem_config)
        tt_result = torch.complex(
            ttnn_ops.ttnn_tensor_to_torch(tt_result.real).to(torch.float),
            ttnn_ops.ttnn_tensor_to_torch(tt_result.imag).to(torch.float),
        )

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success, f"{pcc_value}"


test_sweep_args = [
    (
        [(224, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        14112040,
    ),
    (
        [(6, 160, 64)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        9456908,
    ),
    (
        [(3, 2, 192, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        11871267,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_polar(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_polar_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
