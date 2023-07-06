#include <cstdint>

#include "dataflow_kernel_api.h"

void kernel_main() {
    constexpr std::uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);
    std::uint32_t src_addr_base = dataflow::get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = dataflow::get_arg_val<uint32_t>(1);

    constexpr bool IS_DRAM = true;
    const uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = dataflow::get_tile_size(cb_id);
    dataflow::InterleavedAddrGen<IS_DRAM> src_addrgen = {
        .bank_base_address = src_addr_base,
        .page_size = page_size,
    };

    // read tiles from src to CB
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = dataflow::get_noc_addr(i, src_addrgen);

        dataflow::cb_reserve_back(cb_id, ublock_size_tiles);
        uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id);
        dataflow::noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);

        dataflow::noc_async_read_barrier();

        dataflow::cb_push_back(cb_id, ublock_size_tiles);
    }
}
