// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "common/constants.hpp"
#include "dataflow_api.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr bool output_is_dram = get_compile_time_arg_val(2) == 1;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t end_id = start_id + num_tiles;

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = dst_addr, .page_size = get_tile_size(dst_cb_id), .data_format = get_dataformat(dst_cb_id)};

    uint32_t max_uint = 4294967295;
    float random_range = f2u_to.f - f2u_from.f;

    cb_reserve_back(dst_cb_id, 1);
    uint32_t dst_cb_write_ptr = get_write_ptr(dst_cb_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(intermed_cb_id, 1);

        uint32_t intermed_cb_read_ptr = get_read_ptr(intermed_cb_id);

        uint32_t *intermed_cb_addr = reinterpret_cast<uint32_t *>(intermed_cb_read_ptr);
        uint8_t *dst_cb_addr = reinterpret_cast<uint8_t *>(dst_cb_write_ptr);

        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                uint32_t rand_uint32 = *intermed_cb_addr;
                float rand_float = static_cast<float>(rand_uint32) / max_uint;
                // The hardware PRNG is not uniformly distribute.
                // Generated rand_floats in range [0, 0.5] has higher ratio compared to (0.5, 1).
                // I *2 rand_float < 0.5 to make it more uniform.
                if (rand_float < 0.5f) {
                    rand_float *= 2;
                }
                rand_float = rand_float * random_range + f2u_from.f;

#ifdef OUTPUT_DTYPE_FLOAT32
                *(float *)dst_cb_addr = rand_float;
                dst_cb_addr += 4;
#endif
#ifdef OUTPUT_DTYPE_BFLOAT16
                uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(&rand_float) + 1;
                *(uint16_t *)dst_cb_addr = *uint16_ptr;
                dst_cb_addr += 2;
#endif
                intermed_cb_addr += 1;
            }
        }
        cb_pop_front(intermed_cb_id, 1);

        noc_async_write_tile(i, output_addrg, dst_cb_write_ptr);
        noc_async_write_barrier();
    }
}
