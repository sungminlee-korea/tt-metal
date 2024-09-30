// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/bcast.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(2);

    init_bcast<ELWMUL, BroadcastType::SCALAR>(tt::CB::cb_2, tt::CB::cb_0);
    cb_wait_front(tt::CB::cb_0, onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            cb_wait_front(tt::CB::cb_2, onetile);
            ACQ();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CB::cb_2, tt::CB::cb_0, 0, 0, 0);
            pack_tile(0, tt::CB::cb_16);
            cb_push_back(tt::CB::cb_16, onetile);
            cb_pop_front(tt::CB::cb_2, onetile);
            REL();
        }

        if (has_other_grad) {
            cb_wait_front(tt::CB::cb_1, onetile);
            ACQ();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CB::cb_1, tt::CB::cb_0, 0, 0, 0);
            pack_tile(0, tt::CB::cb_17);
            cb_push_back(tt::CB::cb_17, onetile);
            cb_pop_front(tt::CB::cb_1, onetile);
            REL();
        }
    }
}
}  // namespace NAMESPACE
