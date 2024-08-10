// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t input_dram_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb0_id = 0;

    const uint32_t cb0_page_size = get_tile_size(cb0_id);
    const auto cb0_data_format = get_dataformat(cb0_id);
    /* TODO: fill this section
    const InterleavedAddrGenFast<true> input_addrg = {
        .bank_base_address = , .page_size = , .data_format = };
        */

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb0_id, 1);
        const auto cb0_l1_addr = get_write_ptr(cb0_id);

        // TODO: read tile using input_addrg

        cb_push_back(cb0_id, 1);
    }
}
