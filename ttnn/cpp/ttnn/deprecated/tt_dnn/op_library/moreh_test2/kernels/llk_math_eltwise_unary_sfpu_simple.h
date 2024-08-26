// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_simple.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_simple_tile_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::simple_tile_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_simple_tile(uint dst_index, uint bit_index = 0) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_simple_tile<APPROXIMATE, 8>, dst_index, (int)VectorMode::RC, bit_index);
}

}  // namespace ckernel
