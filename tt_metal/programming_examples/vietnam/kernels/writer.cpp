
#include "dataflow_api.h"
#include "debug/dprint.h"
void print_tile(uint32_t addr) {
  auto ptr = reinterpret_cast<uint16_t *>(addr);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      DPRINT << BF16(ptr[r * 32 + c]) << " ";
    }
    DPRINT << ENDL();
  }

  DPRINT << ENDL();
}

void kernel_main() {
  DPRINT << "Hello, World! I'm reader kernel" << ENDL();
  DPRINT << "Writer noc_idx is " << uint32_t(noc_index) << ENDL();

  // TODO: get arguments
  uint32_t arg = 0;
  const auto dram_buffer1_addr = get_arg_val<uint32_t>(arg++);
  const auto cb1_id = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);

  constexpr bool dram_buffer1_is_dram = get_compile_time_arg_val(0) == 1;

  //TODO: make addr generator for dram buffer1
  const uint32_t cb1_page_size = get_tile_size(cb1_id);
  const auto cb1_data_format = get_dataformat(cb1_id);
  const InterleavedAddrGenFast<dram_buffer1_is_dram> dram_buffer1_addrg = {
      .bank_base_address = dram_buffer1_addr,
      .page_size = cb1_page_size,
      .data_format = cb1_data_format};

  // TODO: write tiles.
  for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    cb_wait_front(cb1_id, 1);
    const auto cb1_l1_addr = get_read_ptr(cb1_id);
    noc_async_write_tile(tile_idx, dram_buffer1_addrg, cb1_l1_addr);
    noc_async_write_barrier();
    cb_pop_front(cb1_id, 1);
  }

  DPRINT << "Writer END" << ENDL();
}
