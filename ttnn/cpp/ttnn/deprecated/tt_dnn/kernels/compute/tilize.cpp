// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

//#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    //UNPACK(( DPRINT << "Block count=" << uint32_t(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt << ENDL() ));
    tilize_init(tt::CB::cb_0, per_core_block_tile_cnt, tt::CB::cb_16);

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(tt::CB::cb_0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::cb_16, per_core_block_tile_cnt);

        tilize_block(tt::CB::cb_0, per_core_block_tile_cnt, tt::CB::cb_16);

        cb_push_back(tt::CB::cb_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::cb_0, per_core_block_tile_cnt);
    }
}
}
