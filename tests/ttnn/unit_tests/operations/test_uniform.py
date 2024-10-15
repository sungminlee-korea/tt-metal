# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import time
import pytest
import numpy as np
import ttnn
from collections import Counter
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
)
from models.utility_functions import skip_for_grayskull
from enum import Enum


class TestMode(Enum):
    VALIDATE = 0
    BENCHMARK = 1


def check_torch_uniform_bfloat16():
    input = torch.zeros(10, 10, dtype=torch.bfloat16).uniform_(2.1, 2.11)
    logger.info(input)


# ttnn is faster than torch if the tensor is big enough: [1024, 1024]
def benchmark_uniform(cpu_input, npu_input, rand_from, rand_to):
    iter_num = 10

    cpu_total_time = 0
    for i in range(iter_num + 1):
        cpu_start_time = time.time_ns()
        cpu_input.uniform_(rand_from, rand_to)
        cpu_end_time = time.time_ns()
        if i > 0:
            cpu_total_time += cpu_end_time - cpu_start_time
    logger.info(f"CPU avg time: {cpu_total_time / iter_num}ns")

    npu_total_time = 0
    for i in range(iter_num + 1):
        npu_start_time = time.time_ns()
        ttnn.uniform(npu_input, rand_from, rand_to)
        npu_end_time = time.time_ns()
        if i > 0:
            npu_total_time += npu_end_time - npu_start_time
    logger.info(f"NPU avg time: {npu_total_time / iter_num}ns")


def validate_uniform(npu_input, shape, rand_from, rand_to, dtype, compute_kernel_config):
    ttnn.uniform(npu_input, rand_from, rand_to, compute_kernel_config=compute_kernel_config)
    tt_input = to_cpu(npu_input, shape)
    elem_cnt = Counter(tt_input.flatten().tolist())

    expected_mean, expected_var = (rand_from + rand_to) / 2, pow(rand_to - rand_from, 2) / 12
    npu_mean, npu_var = torch.mean(tt_input).item(), torch.var(tt_input).item()
    min_val, max_val = torch.min(tt_input).item(), torch.max(tt_input).item()

    logger.info(f"Distinct elements: {len(elem_cnt.keys())}")
    logger.info(f"Expected mean: {expected_mean}, NPU mean: {npu_mean}")
    logger.info(f"Expected var: {expected_var}, NPU var: {npu_var}")

    bfloat16_ep = 0.015

    """
    Random bfloat16 is converted from random float. As 16 bits are truncated, the generated number might be smaller/bigger than from/to.
    (even torch can't handle that case, check check_torch_uniform_bfloat16() function). I use bfloat16_ep is used to avoid asserting fail.
    """
    if dtype == "bfloat16":
        assert rand_from - bfloat16_ep <= min_val and max_val < rand_to + bfloat16_ep
    else:
        assert rand_from <= min_val and max_val < rand_to
    assert np.allclose(npu_mean, expected_mean, rtol=0.5)
    assert np.allclose(npu_var, expected_var, rtol=0.5)


def get_lib_dtype(lib, dtype):
    """Maps dtype to corresponding library dtype."""
    dtype_map = {
        "bfloat16": lib.bfloat16,
        "float32": lib.float32,
    }
    return dtype_map.get(dtype, None)


def run_uniform(shape, rand_range, dtype, device, compute_kernel_options=None, mode=TestMode.VALIDATE):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    rand_from, rand_to = rand_range[0], rand_range[1]
    cpu_input = torch.ones(shape, dtype=get_lib_dtype(torch, dtype))
    npu_input = to_npu(cpu_input, device, npu_dtype=get_lib_dtype(ttnn, dtype))

    if mode == TestMode.BENCHMARK:
        benchmark_uniform(cpu_input=cpu_input, npu_input=npu_input, rand_from=rand_from, rand_to=rand_to)
    else:
        validate_uniform(
            npu_input=npu_input,
            shape=shape,
            rand_from=rand_from,
            rand_to=rand_to,
            dtype=dtype,
            compute_kernel_config=compute_kernel_config,
        )


# fmt: off
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("shape",
    [
        [100, 100],
        [1, 512, 2, 256],
        [512, 512],
        [1024, 1024],
    ],
)
@pytest.mark.parametrize("rand_range",
    [
        [2.1, 9],
        [-5.1, 1.2] # negative float range
    ]
)
@pytest.mark.parametrize("dtype",
    [
        "bfloat16",
        "float32"
    ]
)
# fmt: on
def test_uniform(shape, rand_range, dtype, device):
    torch.manual_seed(0)
    run_uniform(shape, rand_range, dtype, device)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [[2, 32, 32, 16]],
)
@pytest.mark.parametrize("rand_range", [[-3, 4]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_uniform_callback(shape, rand_range, dtype, device, use_program_cache):
    torch.manual_seed(0)
    for i in range(2):
        run_uniform(shape, rand_range, dtype, device)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 2, 4, 70, 40]],
)
@pytest.mark.parametrize("rand_range", [[0, 1]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_uniform_with_compute_kernel_options(shape, rand_range, dtype, device, compute_kernel_options):
    torch.manual_seed(0)
    run_uniform(shape, rand_range, dtype, device, compute_kernel_options)
