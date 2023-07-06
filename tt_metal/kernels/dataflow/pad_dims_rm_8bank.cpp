#include <stdint.h>
#include "dataflow_kernel_api.h"

uint64_t round_down_32(uint64_t a){
    return (a >> 5) << 5;
}

void kernel_main() {

    constexpr uint32_t alignment = 32;

    const uint32_t src_addr                 = dataflow::get_arg_val<uint32_t>(0);
    const uint32_t dst_addr                 = dataflow::get_arg_val<uint32_t>(1);
    const uint32_t num_unpadded_W           = dataflow::get_arg_val<uint32_t>(2);
    const uint32_t num_total_W              = dataflow::get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_Z           = dataflow::get_arg_val<uint32_t>(4);
    const uint32_t num_total_Z              = dataflow::get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_Y           = dataflow::get_arg_val<uint32_t>(6);
    const uint32_t num_total_Y              = dataflow::get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_X           = dataflow::get_arg_val<uint32_t>(8);
    const uint32_t num_total_X              = dataflow::get_arg_val<uint32_t>(9);
    const uint32_t unpadded_X_size          = dataflow::get_arg_val<uint32_t>(10);
    const uint32_t padded_X_size            = dataflow::get_arg_val<uint32_t>(11);
    const uint32_t padded_X_diff_size       = dataflow::get_arg_val<uint32_t>(12);
    const uint32_t pad_value                = dataflow::get_arg_val<uint32_t>(13);
    const uint32_t cache_buffer_l1_addr     = dataflow::get_arg_val<uint32_t>(14);
    const uint32_t src_buffer_l1_addr       = dataflow::get_arg_val<uint32_t>(15);
    const uint32_t dst_buffer_l1_addr       = dataflow::get_arg_val<uint32_t>(16);


    std::uint32_t* cache_buffer = (uint32_t*)(cache_buffer_l1_addr);
    std::uint32_t* src_buffer = (uint32_t*)(src_buffer_l1_addr);
    std::uint32_t* dst_buffer = (uint32_t*)(dst_buffer_l1_addr);


    #define src_stick_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (src_stick_size_is_pow2)
    const uint32_t src_log_base_2_of_page_size = dataflow::get_arg_val<uint32_t>(17);
    const dataflow::InterleavedPow2AddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .log_base_2_of_page_size = src_log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const dataflow::InterleavedAddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .page_size = unpadded_X_size
    };
    #endif

    #define dst_stick_size_is_pow2 get_compile_time_arg_val(1) == 1
    #if (dst_stick_size_is_pow2)
    const uint32_t dst_log_base_2_of_page_size = dataflow::get_arg_val<uint32_t>(18);
    const dataflow::InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = dst_log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const dataflow::InterleavedAddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .page_size = padded_X_size
    };
    #endif

    uint32_t src_stick_id = 0;
    uint32_t dst_stick_id = 0;
    uint32_t l1_cache_addr = cache_buffer_l1_addr;
    for (uint32_t w = 0; w < num_total_W; w++) {
        for (uint32_t z = 0; z < num_total_Z; z++) {
            for (uint32_t y = 0; y < num_total_Y; y++) {
                uint64_t dst_noc_addr = dataflow::get_noc_addr(dst_stick_id, s1);
                uint64_t dst_round_down_addr = round_down_32(dst_noc_addr);
                uint32_t dst_diff_bytes = dst_noc_addr - dst_round_down_addr;
                uint32_t dst_buffer_l1_addr_real = dst_buffer_l1_addr + dst_diff_bytes;
                volatile std::uint32_t* dst = (volatile uint32_t*)(dst_buffer_l1_addr);
                // Copy from cache to tmp buffer
                volatile std::uint32_t* cache = (volatile uint32_t*)(l1_cache_addr);
                for(uint32_t z = 0; z < dst_diff_bytes / 4; z++) {
                    dst[z] = cache[z];
                }
                // if (dst_diff_bytes != 0){
                //     dataflow::noc_async_read(dst_round_down_addr, dst_buffer_l1_addr, dst_diff_bytes);
                //     dataflow::noc_async_read_barrier();
                // }
                dst = (volatile uint32_t*)(dst_buffer_l1_addr_real);

                if (y >= num_unpadded_Y || z >= num_unpadded_Z || w >= num_unpadded_W) {
                    // pad the tile by reading values from zero buffer in L1
                    for(uint32_t z = 0; z < padded_X_size / 4; z++) {
                        dst[z] = pad_value;
                    }
                } else {
                    uint64_t src_noc_addr = dataflow::get_noc_addr(
                        src_stick_id, s0);

                    // Read from DRAM to tmp buffer
                    uint64_t src_round_down_addr = round_down_32(src_noc_addr);
                    uint64_t src_diff_bytes = src_noc_addr - src_round_down_addr;
                    dataflow::noc_async_read(src_round_down_addr, src_buffer_l1_addr, unpadded_X_size + src_diff_bytes);
                    volatile std::uint32_t* dst_pad = (volatile uint32_t*)(dst_buffer_l1_addr_real + unpadded_X_size);
                    // Pad Columns first
                    for(uint32_t z = 0; z < padded_X_diff_size / 4; z++) {
                        dst_pad[z] = pad_value;
                    }

                    // Block before copying data from tmp to cb buffer
                    dataflow::noc_async_read_barrier();
                    volatile std::uint32_t* data_buffer = (volatile uint32_t*)(src_buffer_l1_addr + src_diff_bytes);
                    for(uint32_t z = 0; z < unpadded_X_size / 4; z++) {
                        dst[z] = data_buffer[z];
                    }

                    src_stick_id++;
                }
                dataflow::noc_async_write(dst_buffer_l1_addr, dst_round_down_addr, padded_X_size + dst_diff_bytes);
                // Copy from tmp to cache
                uint64_t end_noc_addr = dst_noc_addr + padded_X_size;
                uint64_t end_round_down_addr = round_down_32(end_noc_addr);
                uint32_t cache_to_write = end_noc_addr - end_round_down_addr;
                dst = (volatile uint32_t*)(dst_buffer_l1_addr_real + padded_X_size - cache_to_write);
                for(uint32_t z = 0; z < (cache_to_write) / 4; z++) {
                    cache[z] = dst[z];
                }
                dst_stick_id++;
                if (dst_stick_id & 7) {
                    l1_cache_addr += alignment;
                } else {
                    l1_cache_addr = cache_buffer_l1_addr;
                }
                dataflow::noc_async_write_barrier();

            }
        }
    }
}
