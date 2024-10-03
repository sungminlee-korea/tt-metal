// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

typedef union {
    float f;
    uint32_t u;
} value;

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t index_addr = get_arg_val<uint32_t>(1);
    uint32_t fill_value = get_arg_val<uint32_t>(2);
    uint32_t input_page_size = get_arg_val<uint32_t>(3);
    uint32_t index_page_size = get_arg_val<uint32_t>(4);
    uint32_t stride = get_arg_val<uint32_t>(5);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t src_index_id = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1;

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = input_addr, .page_size = input_page_size};

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = index_addr, .page_size = index_page_size};

    value val;
    val.u = fill_value;

    cb_reserve_back(src_cb_id, onetile);

    uint32_t write_addr = get_read_ptr(src_cb_id);
    // uint32_t end_id = start_id + num_tiles;
    auto ptr = reinterpret_cast<uint32_t *>(write_addr);
    for (uint32_t i = 0; i < 1024; i+=stride) {
        ptr[i] = fill_value;
    }

    cb_push_back(src_cb_id, onetile);
}
