#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(1);
    std::uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles_per_cb      = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id             = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);

    uint32_t block_size_bytes = get_tile_size(cb_id) * block_size_tiles; 

    for (uint32_t i = 0; i < num_tiles_per_cb; i += block_size_tiles) {
        std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr); 

        cb_wait_front(cb_id, block_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id);

        noc_async_write(l1_read_addr, dram_buffer_dst_noc_addr, block_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id, block_size_tiles);
        dram_buffer_dst_addr += block_size_bytes;
    }
}