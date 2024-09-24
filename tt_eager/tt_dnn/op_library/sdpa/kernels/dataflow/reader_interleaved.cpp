// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
     return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
 }

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t num_cores = get_compile_time_arg_val(9);

    const uint32_t q_addr  = get_arg_val<uint32_t>(0);
    const uint32_t k_addr  = get_arg_val<uint32_t>(1);
    const uint32_t v_addr         = get_arg_val<uint32_t>(2);
    const uint32_t core_id    = get_arg_val<uint32_t>(4);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(5);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(6);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(7);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(8);
    const uint32_t local_q_start = get_arg_val<uint32_t>(9);
    const uint32_t local_q_end = get_arg_val<uint32_t>(10);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;


    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);


    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr,
        .page_size = q_tile_bytes,
        .data_format = q_data_format
    };

    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr,
        .page_size = k_tile_bytes,
        .data_format = k_data_format
    };

    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr,
        .page_size = v_tile_bytes,
        .data_format = v_data_format
    };

    uint32_t q_tile_id = 0;
    uint32_t k_tile_id = 0;
    uint32_t v_tile_id = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        const uint32_t q_batch_offset = nb * NQH * St * DHt;
        const uint32_t k_batch_offset = nb * NKH * St * DHt;
        const uint32_t v_batch_offset = nb * NKH * St * DHt;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk = local_q_start + q_iter;

                uint32_t q_head_offset = nq * St * DHt;
                uint32_t q_chunk_offset = q_chunk * Sq_chunk_t * DHt;
                q_tile_id = q_batch_offset + q_head_offset + q_chunk_offset;

                // Read Q chunk
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                uint32_t q_write_ptr = get_write_ptr(cb_q_in);

                for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
                    noc_async_read_tile(q_tile_id, q_reader, q_write_ptr);
                    q_tile_id += 1;
                    q_write_ptr += q_tile_bytes;
                }
                noc_async_read_barrier();

                cb_push_back(cb_q_in, q_chunk_tiles);

                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                    const uint32_t k_start_tile_id = k_batch_offset + k_chunk * Sk_chunk_t * DHt;

                    // Read K chunk transposed
                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                    uint32_t k_write_ptr = get_write_ptr(cb_k_in);
                    for (uint32_t col = 0; col < DHt; ++col) {
                        k_tile_id = k_start_tile_id + col;
                        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                            noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                            k_tile_id += DHt;
                            k_write_ptr += k_tile_bytes;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_k_in, k_chunk_tiles);

                    v_tile_id = v_batch_offset + k_chunk * Sk_chunk_t * DHt;
                    // Read V chunk
                    cb_reserve_back(cb_v_in, k_chunk_tiles);
                    uint32_t v_write_ptr = get_write_ptr(cb_v_in);
                    for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
                        noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
                        v_tile_id += 1;
                        v_write_ptr += v_tile_bytes;

                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_v_in, k_chunk_tiles);
                }
            }
        }
    }
}
