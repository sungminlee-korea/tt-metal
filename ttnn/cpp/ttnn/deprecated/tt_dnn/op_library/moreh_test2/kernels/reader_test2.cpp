// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    // compile-time args
    constexpr bool input_is_dram = (get_compile_time_arg_val(0) == 1);
    constexpr bool bitmask_is_dram = (get_compile_time_arg_val(1) == 1);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);

    // runtime args
    ArgFetcher arg_fetcher;
    const auto input_addr{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto bitmask_addr{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto num_input_tiles{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto input_start_id{arg_fetcher.get_next_arg_val<uint32_t>()};

    constexpr uint32_t onetile{1};
    constexpr uint32_t cb_id_in0{0};  // input (bf16)
    constexpr uint32_t cb_id_in1{1};  // input2 (uint8)
    constexpr uint32_t cb_id_in2{2};  // (bf16)

    uint32_t input_tile_bytes{static_cast<uint32_t>(get_tile_size(cb_id_in0))};
    const auto input_data_format{get_dataformat(cb_id_in0)};
    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    uint32_t bitmask_tile_bytes{static_cast<uint32_t>(get_tile_size(cb_id_in1))};
    const auto bitmask_data_format{get_dataformat(cb_id_in1)};
    const InterleavedAddrGenFast<bitmask_is_dram> bitmask_addrg = {
        .bank_base_address = bitmask_addr, .page_size = bitmask_tile_bytes, .data_format = bitmask_data_format};

    // one
    Scalar s;
    s.f = 1.0f;
    fill_cb_with_value(cb_id_in2, s.u);

    uint32_t l1_write_addr;
    uint32_t l1_write_addr2;
    for (uint32_t i = 0; i < num_input_tiles; ++i) {
        uint32_t read_input_tile_id{input_start_id + i};
        // read input tile
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(read_input_tile_id, input_addrg, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);

        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr2 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(read_input_tile_id, bitmask_addrg, l1_write_addr2);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, onetile);

        // TODO: kernel debug print avoids race condition
        // DPRINT << "READER" << read_input_tile_id << " " << l1_write_addr << " " << l1_write_addr2 << " "
        //        << num_input_tiles << "\n";
    }
}
