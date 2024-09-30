// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {


    uint32_t NHtWt = get_arg_val<uint32_t>(0);

    transpose_wh_init(tt::CB::cb_0);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(tt::CB::cb_0, 1);
        cb_reserve_back(tt::CB::cb_16, 1);

        acquire_dst(tt::DstMode::Half);
        transpose_wh_tile(tt::CB::cb_0, 0, 0);
        pack_tile(0, tt::CB::cb_16);
        release_dst(tt::DstMode::Half);

        cb_push_back(tt::CB::cb_16, 1);
        cb_pop_front(tt::CB::cb_0, 1);
    }
}
}
