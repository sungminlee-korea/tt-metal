// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void moreh_mul_sub()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat mulA = dst_reg[0];
        vFloat mulB = dst_reg[32];
        vFloat sub = dst_reg[64];
        vFloat res = sub - mulA * mulB;
        dst_reg[0] = res;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
