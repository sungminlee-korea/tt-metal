"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import pytest
import sys
import torch
import tt_lib as ttl
from pathlib import Path
from functools import partial

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")



from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0, is_wormhole_b0

shapes_mm = [
    # Single core (won't be hit after padding is added for multicast)
    [[1, 1, 32, 32], [1, 1, 32, 32]],
    # Multi core (2% math util)
    [[1, 2, 320, 1024], [1, 1, 1024, 384]],
    # Multi core reuse (25% math util)
    [[1, 2, 512, 1024], [1, 1, 1024, 512]],
    # Multi core reuse multicast in0/in1 (25% math util)
    [[1, 2, 4608, 1024], [1, 1, 1024, 6144]],
    # Multi core reuse multicast in0 (25% math util)
    [[1, 2, 512, 1024], [1, 1, 1024, 6144]],
    # Multi core reuse multicast in1 (25% math util)
    [[1, 2, 4608, 1024], [1, 1, 1024, 512]],
    # Multi core reuse with padding (?% math util)
    [[1, 2, 480, 1024], [1, 1, 1024, 480]],
    # Multi core reuse multicast in0/in1 with padding (?% math util)
    [[1, 2, 4576, 1024], [1, 1, 1024, 6112]],
    [[1, 2, 4416, 1024], [1, 1, 1024, 6048]],
    # Multi core reuse multicast in0 with padding (?% math util)
    [[1, 2, 480, 1024], [1, 1, 1024, 6112]],
    [[1, 2, 320, 1024], [1, 1, 1024, 6048]],
    # Multi core reuse multicast in1 with padding (?% math util)
    [[1, 2, 4576, 1024], [1, 1, 1024, 480]],
    [[1, 2, 4416, 1024], [1, 1, 1024, 320]],
]

if is_wormhole_b0():
    del shapes_mm[1:]

@pytest.mark.parametrize(
    "input_shapes",
    shapes_mm
)
@pytest.mark.parametrize("device_id", (0,))
@pytest.mark.parametrize(
    "dtype", (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B)
)
def test_run_matmul_test(input_shapes, device_id, dtype, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "matmul",
        input_shapes,
        datagen_func,
        comparison_func,
        device_id,
        {
            "dtype": [dtype, dtype],
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            "buffer_type": [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM],
            "output_mem_config": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        },
    )

shapes_bmm = [
    # Single core (won't be hit after padding is added for multicast)
    [[1, 1, 32, 32], [1, 1, 32, 32]],
    # Multi core (2% math util)
    [[1, 2, 320, 1024], [1, 2, 1024, 384]],
    # Multi core reuse (25% math util)
    [[1, 2, 512, 1024], [1, 2, 1024, 512]],
    # Multi core reuse multicast in0/in1 (25% math util)
    [[1, 2, 4608, 1024], [1, 2, 1024, 6144]],
    # Multi core reuse multicast in0 (25% math util)
    [[1, 2, 512, 1024], [1, 2, 1024, 6144]],
    # Multi core reuse multicast in1 (25% math util)
    [[1, 2, 4608, 1024], [1, 2, 1024, 512]],
    # Multi core reuse with padding (?% math util)
    [[1, 2, 480, 1024], [1, 2, 1024, 480]],
    # Multi core reuse multicast in0/in1 with padding (?% math util)
    [[1, 2, 4576, 1024], [1, 2, 1024, 6112]],
    [[1, 2, 4416, 1024], [1, 2, 1024, 6048]],
    # Multi core reuse multicast in0 with padding (?% math util)
    [[1, 2, 480, 1024], [1, 2, 1024, 6112]],
    [[1, 2, 320, 1024], [1, 2, 1024, 6048]],
    # Multi core reuse multicast in1 with padding (?% math util)
    [[1, 2, 4576, 1024], [1, 2, 1024, 480]],
    [[1, 2, 4416, 1024], [1, 2, 1024, 320]],
]
if is_wormhole_b0():
    del shapes_bmm[1:]

@pytest.mark.parametrize(
    "input_shapes,",
    shapes_bmm
)
@pytest.mark.parametrize("device_id", (0,))
@pytest.mark.parametrize(
    "dtype", (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B)
)
def test_run_bmm_test(input_shapes, device_id, dtype, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bmm",
        input_shapes,
        datagen_func,
        comparison_func,
        device_id,
        {
            "dtype": [dtype, dtype],
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            "buffer_type": [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM],
            "output_mem_config": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        },
    )
