// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    DPRINT << "Start tilize_init" << ENDL();
#ifndef SHORT_INIT
    tilize_init(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
#else
    unary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);
    tilize_init_short(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
#endif
    DPRINT << "End tilize_init and start block" << ENDL();

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        DPRINT << "Loop no. " << b << ENDL();
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);

        // tilize_block(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
        unpack_tilize_block(tt::CB::c_in0, per_core_block_tile_cnt);
        for (uint32_t tile = 0; tile < per_core_block_tile_cnt; tile++) {
            acquire_dst(tt::DstMode::Half);
            copy_tile(tt::CB::c_in0, 0, 0);
            pack_tile(0, tt::CB::c_out0, 0);
            release_dst(tt::DstMode::Half);
        }

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }
    DPRINT << "End the loop and start uninit" << ENDL();

    tilize_uninit(tt::CB::c_in0, tt::CB::c_out0);
    DPRINT << "End uninit and kernel" << ENDL();
}
}
