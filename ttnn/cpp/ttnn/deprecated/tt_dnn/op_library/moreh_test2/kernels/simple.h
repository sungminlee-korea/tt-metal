// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_simple.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void simple_tile(uint32_t idst, uint32_t bit_index = 0) {
    MATH((llk_math_eltwise_unary_sfpu_simple_tile<APPROX>(idst, bit_index)));
}

ALWI void simple_tile_init() { MATH((llk_math_eltwise_unary_sfpu_simple_tile_init<APPROX>())); }

}  // namespace ckernel
