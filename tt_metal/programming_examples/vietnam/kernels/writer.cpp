
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t arg = 0;
    const auto device_buffer1_addr = get_arg_val<uint32_t>(arg++);
    const auto cb1_id = get_arg_val<uint32_t>(arg++);
    const auto num_tiles = get_arg_val<uint32_t>(arg++);

    constexpr bool device_buffer1_is_dram = get_compile_time_arg_val(0) == 1;

    const uint32_t cb1_page_size = get_tile_size(cb1_id);
    const auto cb1_data_format = get_dataformat(cb1_id);
    const InterleavedAddrGenFast<device_buffer1_is_dram> dram_buffer1_addrg = {
        .bank_base_address = device_buffer1_addr, .page_size = cb1_page_size, .data_format = cb1_data_format};

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // TODO: write tiles.
        cb_wait_front(cb1_id, 1);
        const auto cb1_l1_addr = get_read_ptr(cb1_id);
        noc_async_write_tile(tile_idx, dram_buffer1_addrg, cb1_l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb1_id, 1);
    }

    DPRINT << "Writer END" << ENDL();
}
