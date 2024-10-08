// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
#ifndef SHORT_INIT
    tilize_init(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
#else
    unary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);
    tilize_init_short(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
#endif

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);

        tilize_block(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }

    tilize_uninit(tt::CB::c_in0, tt::CB::c_out0);

}
}
