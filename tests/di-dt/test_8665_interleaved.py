# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import time

import tt_lib as ttl
from models.utility_functions import comp_pcc, torch2tt_tensor
import torch

from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_pcc


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    ((1024, 4608, 18432, 4, 72, 3, 1, 8, 50000), (1024, 4608, 18432, 4, 72, 3, 1, 1, 20000)),
    ids=["ff1-hang", "ff1-pass"],
)
def test_reproduce_matmul_2d_hang(
    device,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
):
    torch.manual_seed(1234)

    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    in0_dtype = ttl.tensor.DataType.BFLOAT16
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT16

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    b_shape_32 = [1, 1, inner_dim, weights_n // 2]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)
    B_32 = torch.randn(b_shape_32)

    torch_result = torch.matmul(A, B)

    a_t = torch2tt_tensor(A, device, ttl.tensor.Layout.TILE, in0_mem_config, in0_dtype)
    b_t = torch2tt_tensor(B, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)
    b_t_32 = torch2tt_tensor(B_32, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    program_config_32 = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        in0_block_w=in_block_w,  # 3
        out_subblock_h=out_subblock_h,  # 1
        out_subblock_w=out_subblock_w,  # 8
        per_core_M=per_core_M,  # 4
        per_core_N=per_core_N,  # 72
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # out_32 = ttl.operations.primary.matmul(
    #     a_t,
    #     b_t_32,
    #     program_config=program_config_32,
    #     output_mem_config=out_mem_config,
    #     output_dtype=out_dtype,
    #     compute_kernel_config=compute_config,
    # )

    print("First run for a reference output")

    # First run for a reference output
    out = ttl.operations.primary.matmul(
        a_t,
        b_t,
        program_config=program_config,
        output_mem_config=out_mem_config,
        output_dtype=out_dtype,
        compute_kernel_config=compute_config,
    )

    does_pass, output_pcc = comp_pcc(torch_result, torch2tt_tensor(out), 0.99)

    print(f"output pcc is {output_pcc}")

    start_time = time.time()

    # # loop_count iterations to test determinism/hang
    # for i in range(loop_count):
    #     out.deallocate(True)
    #     # out_32.deallocate(True)

    #     # out_32 = ttl.operations.primary.matmul(
    #     #     a_t,
    #     #     b_t_32,
    #     #     program_config=program_config_32,
    #     #     output_mem_config=out_mem_config,
    #     #     output_dtype=out_dtype,
    #     #     compute_kernel_config=compute_config,
    #     # )

    #     out = ttl.operations.primary.matmul(
    #         a_t,
    #         b_t,
    #         program_config=program_config,
    #         output_mem_config=out_mem_config,
    #         output_dtype=out_dtype,
    #         compute_kernel_config=compute_config,
    #     )

    #     if i % 100 == 0:
    #         seconds = time.time() - start_time
    #         print(f"Iteration {i} done, time elapsed from the beginning: {seconds:.2f} seconds")

    # out.deallocate(True)

    # print(f"Iterations with nd output: {nd_output_count}")
