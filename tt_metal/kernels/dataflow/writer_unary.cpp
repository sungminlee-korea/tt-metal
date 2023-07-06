#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t dst_addr  = dataflow::get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = dataflow::get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = dataflow::get_arg_val<uint32_t>(2);
    uint32_t num_tiles = dataflow::get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = dataflow::get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = dataflow::get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);

        dataflow::cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);
        dataflow::noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        dataflow::noc_async_write_barrier();

        dataflow::cb_pop_front(cb_id_out0, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
