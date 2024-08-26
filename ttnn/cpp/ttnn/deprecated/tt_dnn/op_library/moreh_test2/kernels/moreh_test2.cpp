// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simple.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    // runtime args
    ArgFetcher arg_fetcher;
    const uint32_t num_tiles{arg_fetcher.get_next_arg_val<uint32_t>()};
    const uint32_t start_id{arg_fetcher.get_next_arg_val<uint32_t>()};

    constexpr auto cb_in0{tt::CB::c_in0};              // input
    constexpr auto cb_in1{tt::CB::c_in1};              // input2
    constexpr auto cb_in2{tt::CB::c_in2};              // 1.0f
    constexpr auto cb_out0{tt::CB::c_out0};            // output
    constexpr auto cb_intermed0{tt::CB::c_intermed0};  // bf16
    constexpr auto cb_intermed1{tt::CB::c_intermed1};  // bf16
    constexpr uint32_t onetile{1};

    cb_wait_front(cb_in2, onetile);
    binary_op_init_common(cb_in0, cb_intermed0, cb_out0);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t input_tile_id{start_id + i};

        // dst0 = bf16 type input
        tile_regs_acquire();
        cb_wait_front(cb_in0, onetile);
        unpack_reconfig_data_format_srca(cb_in0);
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);

        // dst1 = uint8 type input2
        cb_wait_front(cb_in1, onetile);
        unpack_reconfig_data_format_srca(cb_in1);
        copy_tile_to_dst_init_short(cb_in1);
        copy_tile(cb_in1, 0, 1);

        // SFPU function: write input to dst2
        simple_tile_init();
        simple_tile(0, 0);
// template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
// inline void calculate_simple_tile(uint bit_index) {
// #pragma GCC unroll 0
//     for (int d = 0; d < ITERATIONS; d++) {
//         vFloat input = dst_reg[0];
//         vUInt mask = dst_reg[32];
//         v_if (mask == 0) {
//             dst_reg[64] = vConst0;
//         }
//         v_else { dst_reg[64] = input; } // always comes here
//         v_endif;
//         dst_reg++;
//     }
// }
        cb_pop_front(cb_in0, onetile);
        cb_pop_front(cb_in1, onetile);
        tile_regs_commit();

        // intermed0 = pack input from dst2
        tile_regs_wait();
        cb_reserve_back(cb_intermed0, onetile);
        pack_reconfig_data_format(cb_intermed0);
        pack_tile(2, cb_intermed0);
        cb_push_back(cb_intermed0, onetile);
        tile_regs_release();

        // output (dst0) = input x 1.0f;
        tile_regs_acquire();
        cb_wait_front(cb_intermed0, onetile);
        unpack_reconfig_data_format(cb_intermed0, cb_in2);
        mul_tiles_bcast_scalar_init_short(cb_intermed0, cb_in2);
        mul_tiles_bcast_scalar(cb_intermed0, cb_in2, 0, 0, 0);
        cb_pop_front(cb_intermed0, onetile);
        tile_regs_commit();

        // pack output tile from dst0
        tile_regs_wait();
        cb_reserve_back(cb_out0, onetile);
        pack_reconfig_data_format(cb_out0);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, onetile);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
