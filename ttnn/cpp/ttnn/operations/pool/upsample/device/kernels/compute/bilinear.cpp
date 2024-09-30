// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"

template<uint32_t in_ntiles_hw, uint32_t in_ntiles_c, uint32_t out_ntiles_c, uint32_t unpA_face_r_dim>
inline void reduce_h_fused(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_ntiles_hwc,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id) {

    cb_reserve_back(out_cb_id, 1);
    tile_regs_acquire();
    cb_wait_front(in_cb_id, 4);
    unpack_tilizeA_B_block(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, 0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/, 2 /* unpack 1 or 2 faces ) */, unpA_face_r_dim);
    for (uint32_t c_i = 0; c_i < in_ntiles_c; ++c_i) {
        reduce_tile_math(c_i,  2 /* reduce 1 or 2 faces */);
    }
    cb_pop_front(in_cb_id, 4);

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<out_ntiles_c>(out_cb_id, 1, 0, 1, 2);  /* pack 1 row (1x16 or 1x32) */
    tile_regs_release();

    cb_push_back(out_cb_id, 1);
}

namespace NAMESPACE{
void MAIN{
    constexpr uint32_t out_cb_id = tt::CB::cb_16;
    constexpr uint32_t in1_cb_id = tt::CB::cb_1;
    constexpr uint32_t bias_cb_id = tt::CB::cb_2;
    constexpr uint32_t in_scalar_cb_id = tt::CB::cb_4;
    constexpr uint32_t in2_cb_id = tt::CB::cb_24;

    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    constexpr uint32_t in_ntiles_hwc = get_compile_time_arg_val(2);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(3);
    constexpr uint32_t out_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(7);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(8);
    constexpr uint32_t num_output_tiles = out_ntiles_c; //* nblocks;

    tilizeA_B_reduce_init<false, true>(in1_cb_id, in_scalar_cb_id, in_ntiles_hwc, out_cb_id, 2, 4);
    pack_untilize_dst_init_short<num_output_tiles>(out_cb_id, 1, 2); /* pack 1 row (1x16 or 1x32) */
    for(uint32_t i = 0; i < nsticks_per_core_by_nblocks; i++){
        cb_wait_front(in_scalar_cb_id, 1);
        reduce_h_fused<in_ntiles_hw, in_ntiles_c, out_ntiles_c, window_size_hw>(in1_cb_id,
        in_scalar_cb_id, in_ntiles_hwc, i, out_cb_id);
        cb_pop_front(in_scalar_cb_id, 1);
    }
} // MAIN
} //NAMESPACE
