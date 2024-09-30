// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"

#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n / 2; ++i) {
        ptr[i] = (val | (val << 16));
    }
    return true;
}

/**
 * Max-pool 2D.
 */
void kernel_main() {
    const uint32_t reader_nindices = get_compile_time_arg_val(0);
    const uint32_t window_h = get_compile_time_arg_val(1);
    const uint32_t window_w = get_compile_time_arg_val(2);

    const int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes, multiple of 32
    const uint32_t in_nbytes_c = get_compile_time_arg_val(4);
    const uint32_t in_nbytes_c_log2 = get_compile_time_arg_val(5);

    // input tensor height / width / channels
    const int32_t in_w = get_compile_time_arg_val(6);
    const uint32_t in_cb_nsticks = get_compile_time_arg_val(7);

    const uint32_t in_c = get_compile_time_arg_val(8);
    const uint32_t nblocks = get_compile_time_arg_val(9);

    const uint32_t split_reader = get_compile_time_arg_val(10);
    const uint32_t reader_id = get_compile_time_arg_val(11);

    // compile time args
    // value of 1 in bf16 in a uin32_t
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(12);

    // static_assert(0 == reader_nindices%2, "reader_nindices must be multiple of 2");

    constexpr uint32_t TILE_SIZE = 32 * 32;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;
    constexpr uint32_t MAX_ELE_PER_REDUCTION = 512;

    constexpr uint32_t in_cb_id = (reader_id == 1) ? tt::CB::cb_1 : tt::CB::cb_0;
    constexpr uint32_t in_shard_cb_id = tt::CB::cb_2;  // local input shard
    constexpr uint32_t in_reader_indices_cb_id = tt::CB::cb_3;
    constexpr uint32_t in_scalar_cb_id = tt::CB::cb_4;
    constexpr uint32_t interm_reduction_cb_id = tt::CB::cb_25;

    constexpr uint32_t ROW_HW = 64;

    // minus infinity for bfp16
    uint16_t minus_inf = 63487;
    // Reduce scalar = 1
    if (reader_id == 0) {
        cb_reserve_back(in_scalar_cb_id, 1);

        uint32_t bf16_one_u16 = bf16_one_u32 >> 16;
        // fill 1 row w/ scalar
        fill_with_val(get_write_ptr(in_scalar_cb_id), ROW_HW, bf16_one_u16);
        // fill interm buffer with minus_inf
        fill_with_val(get_write_ptr(interm_reduction_cb_id), TILE_SIZE * MAX_TILES_PER_REDUCTION, minus_inf);
        cb_push_back(in_scalar_cb_id, 1);
    }

    uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);

    uint32_t in_w_padded = in_w + 2 * pad_w;

    uint32_t npages_to_reserve = nblocks;
    uint32_t num_8_tile_blocks = 1;
    uint32_t read_bytes = in_nbytes_c;
    if (in_nbytes_c > MAX_ELE_PER_REDUCTION) {
        num_8_tile_blocks = in_nbytes_c / MAX_ELE_PER_REDUCTION;
        read_bytes = MAX_ELE_PER_REDUCTION;  // for now, pow of 2 channels are only supported.
    }
    uint32_t counter = reader_id;
    uint32_t total_elems_to_reduce = window_h * window_w;
    uint32_t remaining_elems = total_elems_to_reduce % MAX_ROWS_FOR_REDUCTION;
    while (counter < reader_nindices) {
        for (uint32_t j = 0; j < num_8_tile_blocks; j++) {
            for (uint32_t i = 0; i < nblocks; ++i) {
                uint16_t top_left_local_index = reader_indices_ptr[counter];
                uint32_t h_multiples = 0;
                uint32_t processed_rows = 0;
                uint32_t out_l1_write_addr_base = get_write_ptr(in_cb_id);
                uint32_t out_l1_write_addr = out_l1_write_addr_base;
                cb_reserve_back(in_cb_id, npages_to_reserve);
                for (uint32_t h = 0; h < window_h; ++h, h_multiples += in_w_padded) {
                    uint32_t stick_offset = top_left_local_index + h_multiples;
                    uint32_t read_offset =
                        j * MAX_ELE_PER_REDUCTION + in_l1_read_base_addr + (stick_offset << in_nbytes_c_log2);
                    for (uint32_t w = 0; w < window_w; w++) {
                        noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, read_bytes);
                        out_l1_write_addr += read_bytes;
                        read_offset += in_nbytes_c;
                        processed_rows++;
                        if ((processed_rows % MAX_ROWS_FOR_REDUCTION) == 0) {
                            noc_async_read_barrier();
                            cb_push_back(in_cb_id, npages_to_reserve);
                            out_l1_write_addr_base = get_write_ptr(in_cb_id);
                            out_l1_write_addr = out_l1_write_addr_base;
                            cb_reserve_back(in_cb_id, npages_to_reserve);
                            // If next is last chunk, fill whole buffer with -inf.
                            if ((total_elems_to_reduce - processed_rows) < MAX_ROWS_FOR_REDUCTION)
                                fill_with_val(out_l1_write_addr, TILE_SIZE * MAX_TILES_PER_REDUCTION, minus_inf);
                        }
                    }
                }
                if (remaining_elems) {
                    noc_async_read_barrier();
                    cb_push_back(in_cb_id, npages_to_reserve);
                }
            }
        }
        counter++;
        if (split_reader)
            counter++;  // interleave the indices
    }
}  // kernel_main()
