// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

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
    uint32_t start_row_id = get_arg_val<uint32_t>(5);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(6);
    uint32_t rows_to_fill = get_arg_val<uint32_t>(7);
    uint32_t num_rows_to_fill_per_index = get_arg_val<uint32_t>(8);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t index_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t stride = get_compile_time_arg_val(4);
    constexpr bool is_last_dim = get_compile_time_arg_val(5) == 1;
    constexpr bool is_first_dim = get_compile_time_arg_val(6) == 1;

    constexpr uint32_t onetile = 1;

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = input_addr, .page_size = input_page_size};

    const InterleavedAddrGen<true> s1 = {
        .bank_base_address = index_addr, .page_size = index_page_size};

    value val;
    val.u = fill_value;


    // cb_reserve_back(src_cb_id, onetile);
    cb_reserve_back(index_cb_id, onetile);

    // uint32_t src_cb_reader = get_write_ptr(src_cb_id);
    // uint64_t input_noc_addr = get_noc_addr(0, s0);
    // noc_async_read(input_noc_addr, src_cb_reader, input_page_size);
    // noc_async_read_barrier();

    uint32_t index_cb_reader = get_write_ptr(index_cb_id);
    uint64_t index_noc_addr = get_noc_addr(0, s1);
    noc_async_read(index_noc_addr, index_cb_reader, index_page_size);
    noc_async_read_barrier();
    // uint32_t* input_ptr = reinterpret_cast<uint32_t*>(src_cb_reader);
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_cb_reader);
    // DPRINT << index_page_size / 4 << ENDL();

    if (is_last_dim) {
        for (uint32_t row_id = start_row_id; row_id < start_row_id + num_rows_per_core; row_id++) {
            cb_reserve_back(src_cb_id, onetile);
            uint32_t src_cb_reader = get_write_ptr(src_cb_id);
            uint64_t input_noc_addr = get_noc_addr(row_id, s0);
            noc_async_read(input_noc_addr, src_cb_reader, input_page_size);
            noc_async_read_barrier();

            uint32_t* input_ptr = reinterpret_cast<uint32_t*>(src_cb_reader);


            for (uint32_t i = 0;i < index_page_size / 4; i++) {

                uint32_t current_index = index_ptr[i];
                input_ptr[current_index] = fill_value;
            }

            cb_push_back(src_cb_id, onetile);
        }
    } else if (is_first_dim) {




        uint32_t* input_ptr = reinterpret_cast<uint32_t*>(src_cb_reader);


        for (uint32_t i = 0;i < index_page_size / 4; i++) {
            for (int j = 0;j < num_rows_to_fill_per_index;j++) {
                cb_reserve_back(src_cb_id, onetile);
                uint32_t src_cb_reader = get_write_ptr(src_cb_id);
                uint64_t input_noc_addr = get_noc_addr(j + index_ptr[i] * num_rows_to_fill_per_index, s0);
                noc_async_read(input_noc_addr, src_cb_reader, input_page_size);
                noc_async_read_barrier();

                fill_cb_with_value(src_cb_id, fill_value); // Check
                cb_push_back(src_cb_id, onetile);
            }
            uint32_t current_index = index_ptr[i];
            input_ptr[current_index] = fill_value;
        }



    }
    cb_push_back(index_cb_id, onetile);

    // cb_push_back(index_cb_id, onetile);
    // cb_push_back(src_cb_id, onetile);


}
