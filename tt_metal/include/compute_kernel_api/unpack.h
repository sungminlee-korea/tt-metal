// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

/**
 * Helper function to reconfigure unpacker srca and srcb input data formats.
 */
template <bool to_from_int8 = false>
ALWI void unpack_reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    UNPACK(( llk_unpack_reconfig_data_format<to_from_int8, DST_ACCUM_MODE>(srca_new_operand, srcb_new_operand) ));
}

/**
 * Helper function to reconfigure srca/srcb input data formats, only if they differ from existing formats.
*/
template <bool to_from_int8 = false>
ALWI void unpack_reconfig_data_format(const uint32_t srca_old_operand, const uint32_t srca_new_operand, const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    UNPACK(( llk_unpack_reconfig_data_format<to_from_int8, DST_ACCUM_MODE>(srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand) ));
}

/**
 * Helper function to reconfigure unpacker srca input data format.
 */
template <bool to_from_int8 = false>
ALWI void unpack_reconfig_data_format_srca(const uint32_t srca_new_operand) {
    UNPACK(( llk_unpack_reconfig_data_format_srca<to_from_int8, DST_ACCUM_MODE>(srca_new_operand) ));
}

/**
 * Helper function to reconfigure unpacker srca input data format, only if it differs from existing format.
 */
template <bool to_from_int8 = false>
ALWI void unpack_reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    UNPACK(( llk_unpack_reconfig_data_format_srca<to_from_int8, DST_ACCUM_MODE>(srca_old_operand, srca_new_operand) ));
}

/**
 * Helper function to reconfigure unpacker srcb input data format.
 */
template <bool to_from_int8 = false>
ALWI void unpack_reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    UNPACK(( llk_unpack_reconfig_data_format_srcb<to_from_int8, DST_ACCUM_MODE>(srcb_new_operand) ));
}

/**
 * Helper function to reconfigure unpacker srcb input data format, only if it differs from existing format.
 */
template <bool to_from_int8 = false>
ALWI void unpack_reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    UNPACK(( llk_unpack_reconfig_data_format_srcb<to_from_int8, DST_ACCUM_MODE>(srcb_old_operand, srcb_new_operand) ));
}

}
