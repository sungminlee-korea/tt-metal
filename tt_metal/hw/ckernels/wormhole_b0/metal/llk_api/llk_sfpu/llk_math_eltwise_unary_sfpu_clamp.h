// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_3_param.h"
#include "ckernel_sfpu_clamp.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_clamp_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::clamp, APPROXIMATE>(sfpu::clamp_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_clamp(uint dst_index, uint param0, uint param1, uint param2, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_3_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_clamp<APPROXIMATE>,
                                ckernel::sfpu::calculate_clamp<APPROXIMATE>,
                                dst_index, vector_mode, param0, param1, param2);
}

}
