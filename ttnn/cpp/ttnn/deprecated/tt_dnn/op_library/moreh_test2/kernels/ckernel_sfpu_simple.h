// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
void simple_tile_init() {}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_simple_tile(uint bit_index) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input = dst_reg[0];
        vUInt mask = dst_reg[32];
        v_if (mask == 0) {
            dst_reg[64] = vConst0;
        }
        v_else { dst_reg[64] = input; }
        v_endif;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
