# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import tt_lib as ttl
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000


def is_unsupported_case(input_shape, scatter_dim, math_op, mem_config, num_devices, num_links, input_dtype, layout):
    if scatter_dim != 3:
        return True, "Only support for scatter_dim=3 is tested so far"

    elem_size = 2 if input_dtype == ttl.tensor.DataType.BFLOAT16 else 1
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttl.tensor.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # if input_dtype == ttl.tensor.DataType.BFLOAT8_B and tuple(input_shape) == (1, 1, 2048, 1024) and scatter_dim == 3:
    #     return True, "Known failure with bfp8_b data format"

    return False, ""


def run_reduce_scatter_test(
    t3k_device_mesh,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async=False,
    num_iters=1,
):
    if len(t3k_device_mesh.get_device_ids()) != 8:
        pytest.skip("Not T3000!")

    debug = True

    (is_known_failure, message) = is_unsupported_case(
        per_chip_output_shape, scatter_dim, math_op, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    for device_id in t3k_device_mesh.get_device_ids():
        t3k_device_mesh.get_device(device_id).enable_async(enable_async)
    if enable_async:
        logger.info(f"Using Async Mode for Reduce Scatter Op Dispatch")

    # Generate input tensors
    canonical_input_shape = per_chip_output_shape.copy()
    canonical_input_shape[scatter_dim] *= num_devices
    tt_input_tensors = []

    numel = canonical_input_shape[0] * canonical_input_shape[1] * canonical_input_shape[2] * canonical_input_shape[3]
    input_tensors = [
        # torch.rand(canonical_input_shape).bfloat16() if not debug else torch.arange(numel).reshape(canonical_input_shape).bfloat16()
        torch.rand(canonical_input_shape).bfloat16() if not debug else torch.ones(canonical_input_shape).bfloat16()
        for _ in range(num_devices)
    ]

    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(canonical_input_shape).bfloat16()
    for i, canonical_input_tensor in enumerate(input_tensors):
        tt_input_tensors.append(
            ttl.tensor.Tensor(canonical_input_tensor, input_dtype)
            .to(layout)
            .to(t3k_device_mesh.get_device(t3k_device_mesh.get_device_ids()[i]), mem_config)
        )

    assert len(tt_input_tensors) == num_devices

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    # Run the op
    # for i in range(num_iters):
    output_tensor_mesh = ttnn.reduce_scatter(
        input_tensor_mesh,
        scatter_dim=scatter_dim,
        math_op=math_op,
        num_links=num_links,
        memory_config=mem_config,
    )

    for device_id in t3k_device_mesh.get_device_ids():
        ttl.device.Synchronize(t3k_device_mesh.get_device(device_id))
    logger.info(f"Done iteration 0")

    # Compute golden
    # TODO: Make it model how reduce scatter actually works for numerical correctness/ordering
    golden_canonical_out_tensor = torch.zeros(canonical_input_shape).bfloat16()
    for i, t in enumerate(input_tensors):
        golden_canonical_out_tensor = torch.add(golden_canonical_out_tensor, t).bfloat16()

    golden_output_tensors = torch.chunk(golden_canonical_out_tensor, num_devices, scatter_dim)

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    # Compare
    assert len(golden_output_tensors) == len(tt_out_tensors)
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
            if debug:
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(tt_output_tensor.shape[2]):
                            for x in range(tt_output_tensor.shape[3]):
                                if tt_output_tensor[w, z, y, x] != golden_output_tensors[i][w, z, y, x]:
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_output_tensors[i][w, z, y, x]}"
                                    )

        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


# ~2:45 extra time in the current state
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        # (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 2, 256, 32 * 8], 3, ttl.tensor.Layout.TILE),  # Input tensor is (16*32) x (64*32) = 8 * input tensor shape
        ([1, 1, 32, 32 * 8], 3, ttl.tensor.Layout.TILE),
        ([1, 8, 1024, 1024], 3, ttl.tensor.Layout.TILE),
        ([1, 4, 2048, 1024], 3, ttl.tensor.Layout.TILE),
        # # # Has worker slice size warning - defaults to 1x1
        ([1, 1, 128, 8192], 3, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttl.tensor.ReduceOpMath.SUM])
@pytest.mark.parametrize("enable_async", [True])
def test_reduce_scatter_post_commit(
    t3k_device_mesh,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_reduce_scatter_test(
        t3k_device_mesh,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        enable_async=enable_async,
    )
