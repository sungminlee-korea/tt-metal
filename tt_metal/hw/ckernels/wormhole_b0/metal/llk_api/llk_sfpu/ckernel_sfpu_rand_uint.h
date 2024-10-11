#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_uint_init(uint seed) {
    init_prng_seed(seed);
}

template <bool APPROXIMATION_MODE>
inline void rand_uint() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(0, 4, 3, 0);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG0, 5);
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
