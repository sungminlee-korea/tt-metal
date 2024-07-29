
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
  DPRINT << "Reader noc_idx is " << uint32_t(noc_index) << ENDL();

  DPRINT << "noc coords 0 : " << (uint)my_x[0] << "," << (uint)my_y[0]
         << "\tnoc coords 1 : " << (uint)my_x[1] << "," << (uint)my_y[1]
         << ENDL();

  // TODO: get runtime args and compile time args.
  uint32_t arg = 0;
  const auto dram_buffer0_addr = get_arg_val<uint32_t>(arg++);
  const auto cb0_id = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);

  constexpr bool dram_buffer0_is_dram = get_compile_time_arg_val(0) == 1;

  // TODO: make addr generator for dram.
  const uint32_t cb0_page_size = get_tile_size(cb0_id);
  const auto cb0_data_format = get_dataformat(cb0_id);
  const InterleavedAddrGenFast<dram_buffer0_is_dram> input_addrg = {
      .bank_base_address = dram_buffer0_addr,
      .page_size = cb0_page_size,
      .data_format = cb0_data_format};

  // TODO: read tiles.
  for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    cb_reserve_back(cb0_id, 1);
    const auto cb0_l1_addr = get_write_ptr(cb0_id);
    noc_async_read_tile(tile_idx, input_addrg, cb0_l1_addr, 0 /*offset*/);
    noc_async_read_barrier();
    cb_push_back(cb0_id, 1);
  }

  DPRINT << "READER END" << ENDL();
}
