#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"

inline void write_tiles_in_block(uint32_t cb_id_out0,
                        uint32_t block_height_ntiles,
                        uint32_t block_width_ntiles,
                        uint32_t block_start_row_id,
                        uint32_t block_row_offset,
                        uint32_t block_row_size,
                        uint32_t block_row_size_unpadded, // to remove padding from the last block in the row
                        uint32_t num_rows_unpadded,
                        const InterleavedAddrGen<true>& s) {
    constexpr uint32_t TILE_HEIGHT = 32;  // TODO: use common source of truth
    uint32_t block_row_id = block_start_row_id;
    for (uint32_t tile_row_id = 0; tile_row_id < block_height_ntiles; tile_row_id++) {
        // We reserve back an entire row of tiles in a block and issue a bunch of reads
        cb_wait_front(cb_id_out0, block_width_ntiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < TILE_HEIGHT; j++) {
            if (block_row_id >= num_rows_unpadded) {
                break;
            }
            uint64_t dst_noc_addr = get_noc_addr(block_row_id, s, block_row_offset);
            noc_async_write(l1_read_addr, dst_noc_addr, block_row_size_unpadded);
            l1_read_addr += block_row_size;
            block_row_id++;
        } // for tile_nrows
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, block_width_ntiles);
    } // for block_height_ntiles
}
void kernel_main() {

    uint32_t dst_addr = get_arg_val<uint32_t>(0);           // out_dram_addr
    uint32_t num_rows_block = get_arg_val<uint32_t>(1);
    uint32_t block_row_size = get_arg_val<uint32_t>(2);     // in0_block_w * TILE_WIDTH * dtype_nbytes
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t num_blocks_h = get_arg_val<uint32_t>(4);
    uint32_t num_blocks_w = get_arg_val<uint32_t>(5);
    uint32_t output_row_size = get_arg_val<uint32_t>(6);    // output row size bytes
    uint32_t last_block_row_size_unpadded = get_arg_val<uint32_t>(7); // unpadded last block width
    uint32_t num_output_rows_unpadded = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    constexpr uint32_t TILE_HEIGHT = 32;                    // TODO: use common source of truth

    const uint32_t block_width_ntiles = block_row_size >> 6; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    const uint32_t block_height_ntiles = num_rows_block / TILE_HEIGHT;
    uint32_t block_start_row_id = 0;

    // const InterleavedAddrGenFast<true> s = {
    //     .bank_base_address = dst_addr,
    //     .page_size = output_row_size,
    //     .data_format = out_df
    // };
    const InterleavedAddrGen<true> s = {
        .bank_base_address = dst_addr,
        .page_size = output_row_size
    };
    for(uint32_t b = 0; b < batch; ++b) {
        for(uint32_t block_h = 0; block_h < num_blocks_h; block_h++) {
            uint32_t block_row_offset = 0;
            for(uint32_t block_w = 0; block_w < num_blocks_w; block_w++) {
                uint32_t current_block_row_size_unpadded = block_row_size;
                if(block_w == (num_blocks_w - 1)) {
                    current_block_row_size_unpadded = last_block_row_size_unpadded;
                }
                write_tiles_in_block(cb_id_out0,
                        block_height_ntiles,
                        block_width_ntiles,
                        block_start_row_id,
                        block_row_offset,
                        block_row_size,
                        current_block_row_size_unpadded, // padding is only in the last block
                        num_output_rows_unpadded,
                        s);
                block_row_offset += block_row_size;
            } // for num_blocks_w
            block_start_row_id += num_rows_block;
        } // for num_blocks_h
    } // for batch
}
