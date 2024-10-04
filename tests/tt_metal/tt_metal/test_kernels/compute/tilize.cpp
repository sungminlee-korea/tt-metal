// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK(( DPRINT << "======" << ENDL() ));
    for (uint16_t r = 0; r < 32; ++ r) {
        SliceRange sr = SliceRange{
            .h0 = r,
            .h1 = (uint16_t)(r+1),
            .hs = 1,
            .w0 = 0,
            .w1 = 32,
            .ws = 1
        };
        PACK(( DPRINT << (uint32_t)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
    }
    PACK(( DPRINT << "++++++" << ENDL() ));
}

void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    UNPACK (( DPRINT << "UNPACK started compute init" << ENDL() ));
    MATH (( DPRINT << "MATH started compute init" << ENDL() ));
    PACK (( DPRINT << "PACK started compute init" << ENDL() ));
#ifndef SHORT_INIT
    tilize_init(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
#else
    unary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);
    tilize_init_short(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
#endif
    UNPACK (( DPRINT << "UNPACK finished init, started loop" << ENDL() ));
    MATH (( DPRINT << "MATH finished init, started loop" << ENDL() ));
    PACK (( DPRINT << "PACK finished init, started loop" << ENDL() ));

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);

        tilize_block(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);
        // unpack_tilize_block(tt::CB::c_in0, per_core_block_tile_cnt);
        UNPACK (( DPRINT << "UNPACK finished tilization" << ENDL() ));
        // for (uint32_t tile = 0; tile < per_core_block_tile_cnt; tile++) {
        //     acquire_dst(tt::DstMode::Half);
        //     copy_tile(tt::CB::c_in0, 0, 0);
        //     MATH (( DPRINT << "MATH finished copy, round " << tile << ENDL() ));
        //     pack_tile(0, tt::CB::c_out0, 0);
        //     PACK (( DPRINT << "PACK finished pack, round " << tile << ENDL() ));
        //     release_dst(tt::DstMode::Half);
        //     print_full_tile(tt::CB::c_out0, tile, true);
        // }

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }

    tilize_uninit(tt::CB::c_in0, tt::CB::c_out0);

    UNPACK (( DPRINT << "UNPACK finished compute uninit" << ENDL() ));
    MATH (( DPRINT << "MATH finished compute uninit" << ENDL() ));
    PACK (( DPRINT << "PACK finished compute uninit" << ENDL() ));
}
}
