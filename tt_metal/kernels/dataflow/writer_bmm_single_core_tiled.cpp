#include "dataflow_kernel_api.h"

void kernel_main() {

    uint32_t out_addr                   = dataflow::get_arg_val<uint32_t>(0);
    uint32_t out_stride_w               = dataflow::get_arg_val<uint32_t>(2);
    uint32_t out_stride_h               = dataflow::get_arg_val<uint32_t>(3);
    uint32_t out_next_subblock_stride_w = dataflow::get_arg_val<uint32_t>(4);
    uint32_t out_next_subblock_stride_h = dataflow::get_arg_val<uint32_t>(5);
    uint32_t out_subblock_w             = dataflow::get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h             = dataflow::get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count    = dataflow::get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w        = dataflow::get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h        = dataflow::get_arg_val<uint32_t>(10);

    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    constexpr uint32_t tile_size_pow2_exponent = 11;    // == 2^11 = 2048 = 2 * 32 * 32 (assuming dtype = 2 bytes)
    const dataflow::InterleavedPow2AddrGen<true> s = {
        .bank_base_address = out_addr,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };
    const uint32_t tile_size_bytes = dataflow::get_tile_size(out_cb_id);

    uint32_t out_sbh_start_tile_id = 0;
    for(uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
        uint32_t out_sbw_start_tile_id = out_sbh_start_tile_id;
        for(uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
            uint32_t out_sb_row_start_tile_id = out_sbw_start_tile_id;
            // wait for one subblock worth tiles
            dataflow::cb_wait_front(out_cb_id, out_subblock_tile_count);
            uint32_t l1_read_addr = dataflow::get_read_ptr(out_cb_id);
            for(uint32_t h = 0; h < out_subblock_h; ++h) {
                uint32_t out_tile_id = out_sb_row_start_tile_id;
                for(uint32_t w = 0; w < out_subblock_w; ++w) {
                    uint64_t out_tile_noc_addr = dataflow::get_noc_addr(out_tile_id, s);
                    dataflow::noc_async_write(l1_read_addr, out_tile_noc_addr, tile_size_bytes);
                    l1_read_addr += tile_size_bytes;
                    out_tile_id += out_stride_w;
                }
                out_sb_row_start_tile_id += out_stride_h;
            }
            dataflow::noc_async_write_barrier();
            dataflow::cb_pop_front(out_cb_id, out_subblock_tile_count);
            out_sbw_start_tile_id += out_next_subblock_stride_w;
        }
        out_sbh_start_tile_id += out_next_subblock_stride_h;
    }
}
