// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
     return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
 }

void kernel_main() {
    constexpr uint32_t NKH = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t _unused1 = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8); // num chunks in valid_seq_len

    constexpr uint32_t num_cores = get_compile_time_arg_val(9);

    const uint32_t q_addr  = get_arg_val<uint32_t>(0);
    const uint32_t k_addr  = get_arg_val<uint32_t>(1);
    const uint32_t v_addr         = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr         = get_arg_val<uint32_t>(3);
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
    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t valid_seqlen_tiles = k_num_chunks * Sk_chunk_t;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CB::cb_0;
    constexpr uint32_t cb_k_in = tt::CB::cb_1;
    constexpr uint32_t cb_v_in = tt::CB::cb_2;
    constexpr uint32_t cb_mask_in = tt::CB::cb_3;


    constexpr uint32_t onetile = 1;
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr DataFormat mask_data_format = get_dataformat(cb_mask_in);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<k_tile_bytes, num_cores>();

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

    const InterleavedAddrGenFast<is_dram> mask_reader = {
        .bank_base_address = mask_addr,
        .page_size = mask_tile_bytes,
        .data_format = mask_data_format
    };

    uint32_t k_tile_id = 0;
    uint32_t v_tile_id = 0;
    uint32_t mask_tile_id = 0;
    uint32_t barrier_count = 0;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        DPRINT << "nb: " << nb << ENDL();
        const uint32_t q_batch_offset = nb * NKH * St * DHt;
        const uint32_t k_batch_offset = nb * NKH * St * DHt;
        const uint32_t v_batch_offset = nb * NKH * St * DHt;
        const uint32_t mask_batch_offset = nb * NKH * Sq_chunk_t * valid_seqlen_tiles;
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            DPRINT << "nq: " << nq << ENDL();
            // In decode mode, n1 = q_shape[1] is actually the batch dim
            const uint32_t kv_nh_offset = nq * St * DHt;
            const uint32_t mask_nh_offset = nq * Sq_chunk_t * valid_seqlen_tiles;

            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                DPRINT << "q_iter: " << q_iter << ENDL();

                // Q is available because it's a sharded input
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                cb_push_back(cb_q_in, q_chunk_tiles);

                // Loop over k_num_chunks, which is the number of chunks in `valid_seq_len`
                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
                    DPRINT << "k_chunk: " << k_chunk << ENDL();
                    const uint32_t k_chunk_offset = k_chunk * Sk_chunk_t * DHt;
                    const uint32_t mask_chunk_offset = k_chunk * Sk_chunk_t;

                    const uint32_t k_start_tile_id = k_batch_offset + kv_nh_offset + k_chunk_offset;

                    // Read K chunk transposed
                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                    uint32_t k_write_ptr = get_write_ptr(cb_k_in);
                    barrier_count = 0;
                    for (uint32_t col = 0; col < DHt; ++col) {
                        k_tile_id = k_start_tile_id + col;
                        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                            DPRINT << "k_tile_id: " << k_tile_id << ENDL();
                            noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                            k_tile_id += DHt;
                            k_write_ptr += k_tile_bytes;

                            if (++barrier_count == barrier_threshold) {
                                noc_async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_k_in, k_chunk_tiles);


                    // Noncausal, so we always read mask
                    cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                    uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                    barrier_count = 0;
                    mask_tile_id = mask_batch_offset + mask_nh_offset + mask_chunk_offset;
                    for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                        for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                            DPRINT << "mask_tile_id: " << mask_tile_id << ENDL();
                            noc_async_read_tile(mask_tile_id, mask_reader, mask_write_ptr);
                            mask_tile_id += 1;
                            mask_write_ptr += mask_tile_bytes;

                            if (++barrier_count == barrier_threshold) {
                                noc_async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                        // Strid along columns to get to next row
                        mask_tile_id -= Sk_chunk_t;
                        mask_tile_id += valid_seqlen_tiles;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_mask_in, mask_chunk_tiles);




                    v_tile_id = v_batch_offset + kv_nh_offset + k_chunk_offset;
                    // Read V chunk
                    cb_reserve_back(cb_v_in, k_chunk_tiles);
                    uint32_t v_write_ptr = get_write_ptr(cb_v_in);
                    barrier_count = 0;
                    for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
                        DPRINT << "v_tile_id: " << v_tile_id << ENDL();
                        noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
                        v_tile_id += 1;
                        v_write_ptr += v_tile_bytes;

                        if (++barrier_count == barrier_threshold) {
                            noc_async_read_barrier();
                            barrier_count = 0;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_v_in, k_chunk_tiles);
                }
            }
        }
    }
}
