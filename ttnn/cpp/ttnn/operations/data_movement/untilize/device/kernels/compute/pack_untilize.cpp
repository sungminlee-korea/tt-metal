// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    pack_untilize_init<per_core_block_tile_cnt>(tt::CB::cb_0, tt::CB::cb_16);

    for(uint32_t b = 0; b < per_core_block_cnt; ++ b) {
        cb_wait_front(tt::CB::cb_0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::cb_16, per_core_block_tile_cnt);

        pack_untilize_block<per_core_block_tile_cnt>(tt::CB::cb_0, 1, tt::CB::cb_16);

        cb_push_back(tt::CB::cb_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::cb_0, per_core_block_tile_cnt);
    }

    pack_untilize_uninit();
}
}
