// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    binary_op_init_common(tt::CB::cb_0, tt::CB::cb_1);
    bool enable_reload = false;
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        // elemwise-mul
        ACQ();
        cb_wait_front(tt::CB::cb_0, onetile);
        cb_wait_front(tt::CB::cb_1, onetile);

        cb_reserve_back(tt::CB::cb_24, onetile);
        mul_tiles_init();
        // dst0 = cb_0 x cb_1
        mul_tiles(tt::CB::cb_0, tt::CB::cb_1, 0, 0, 0);
        // cb_24 = pack(dst0)
        pack_tile(0, tt::CB::cb_24);
        cb_push_back(tt::CB::cb_24, onetile);

        cb_pop_front(tt::CB::cb_0, onetile);
        cb_pop_front(tt::CB::cb_1, onetile);
        REL();

        // reduce-w
        ACQ();
        if (enable_reload) {
            cb_wait_front(tt::CB::cb_25, onetile);
            copy_tile_to_dst_init_short();
            copy_tile(tt::CB::cb_25, 0, 0);
            cb_pop_front(tt::CB::cb_25, onetile);
        }

        cb_wait_front(tt::CB::cb_24, onetile);
        reduce_init_delta<false>();
        reduce_tile(tt::CB::cb_24, tt::CB::cb_2, 0, 0, 0);
        cb_pop_front(tt::CB::cb_24, onetile);
        reduce_revert_delta();

        if (last_out) {
            cb_reserve_back(tt::CB::cb_16, onetile);
            pack_tile(0, tt::CB::cb_16);
            cb_push_back(tt::CB::cb_16, onetile);
        } else {
            cb_reserve_back(tt::CB::cb_25, onetile);
            pack_tile(0, tt::CB::cb_25);
            cb_push_back(tt::CB::cb_25, onetile);
        }
        REL();
        enable_reload = true;
    }
}
}  // namespace NAMESPACE
