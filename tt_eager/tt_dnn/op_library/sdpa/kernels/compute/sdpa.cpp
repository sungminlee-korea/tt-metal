// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint.h"
#include "debug/assert.h"

#define DEBUG 1

namespace NAMESPACE {
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK( DPRINT << "======" << ENDL() );
    for (uint16_t r = 0; r < 32; ++ r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PACK( DPRINT << (uint)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() );
    }
    PACK( DPRINT << "++++++" << ENDL() );
}

inline void print_cb_details(uint32_t cb_id) {
    UNPACK(DPRINT << "cb_id " << cb_id << ": { "
            << "size: " << cb_interface[cb_id].fifo_size << ", "
            << "limit: " << cb_interface[cb_id].fifo_limit << ", "
            << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
            << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
            << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
            << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
            << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL());
}

float bfloat16_to_float32(uint16_t bfloat16_value) {
    uint32_t sign = (bfloat16_value & 0x8000) << 16;
    uint32_t exponent = (bfloat16_value & 0x7F80) << 16;
    uint32_t mantissa = (bfloat16_value & 0x007F) << 16;
    uint32_t float32_value = sign | exponent | mantissa;
    union FloatIntUnion {
        float f;
        uint32_t i;
    };
    FloatIntUnion u;
    u.i = float32_value;
    return u.f;
    // return *reinterpret_cast<float*>(&float32_value);
}

void print_tile_bfloat16(const uint32_t cb_id) {

    // uint32_t read_addr = get_read_ptr(cb_id);
    uint32_t read_addr = cb_interface[cb_id].fifo_rd_ptr << 4;
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(read_addr);

    // dprint("32x32 Tile (bfloat16 to float32):\n");

    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j = 0; j < 32; ++j) {
            uint16_t bfloat16_value = ptr[i * 32 + j];
            float float_value = bfloat16_to_float32(bfloat16_value);
            // dprint("%.6f ", float_value);
            DPRINT << float_value << " ";
            if (j == 15) {
                DPRINT << "  ";
            }
        }

        DPRINT << ENDL();
        if (i == 15) {
            DPRINT << ENDL();
        }
    }
}


template<PoolType pool_type, ReduceDim reduce_dim, uint32_t in0_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t rows, uint32_t cols>
void reduce_c() {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced
    // DPRINT << "CALLED " << " r " << rows << " c " << cols;
    // called 2 rows 2 col 4
    // reduce_init_delta<false, pool_type, reduce_dim>(pool_type, reduce_dim, in0_cb, scale_cb, out_cb);
    MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(in0_cb, scale_cb) ));
    // MATH(( llk_math_eltwise_unary_datacopy_init<A2D>(0,0, scale_cb) ));
    // UNPACK(( llk_unpack_A_init(0,0, scale_cb) ));
    // unpack_reconfig_data_format(in0_cb, scale_cb);
    // pack_reconfig_data_format(out_cb);

    const uint32_t num_tiles = rows * cols;
    // cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst(tt::DstMode::Half);
        for (uint32_t j = 0; j < cols; j++) {

            MATH(( llk_math_eltwise_binary<ELWMUL, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(0, 0, 0) ));
            UNPACK(( llk_unpack_AB(in0_cb, scale_cb, 0, 0) ));
        }

        cb_reserve_back(out_cb, 1);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }
    reduce_revert_delta<reduce_dim>(out_cb);
}

void matmul_blocks(const uint32_t& in0_cb, const uint32_t& in1_cb, const uint32_t& out_cb, const uint32_t& M, const uint32_t& N, const uint32_t& K, const uint32_t& num_blocks, const uint32_t& in0_num_subblocks, const uint32_t& in1_num_subblocks,
                    const uint32_t& in0_block_w, const uint32_t& subblock_h, const uint32_t& subblock_w, const bool& transpose) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    mm_block_init_short(in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    unpack_reconfig_data_format(in1_cb, in0_cb);
    // cb_wait_front(in1_cb, K * N);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;
    uint32_t in1_index_offset = 0;

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;

            }
            tile_regs_commit();

            cb_reserve_back(out_cb, out_subblock_num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, out_subblock_num_tiles);
            in1_index_offset += in1_subblock * subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_pop_front(in1_cb, K * N);
}

void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(9);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(11);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(15);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(17);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(18);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(20);

    constexpr uint32_t num_cores = get_compile_time_arg_val(21);

    const uint32_t core_id    = get_arg_val<uint32_t>(0);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);


    const uint32_t q_chunks_per_core = local_q_end - local_q_start;


    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_out_im = tt::CB::c_intermed1;
    constexpr uint32_t cb_out_accumulate_im = tt::CB::c_intermed2;
    constexpr uint32_t cb_cur_max = tt::CB::c_intermed3;
    constexpr uint32_t cb_prev_max = tt::CB::c_intermed4;
    constexpr uint32_t cb_cur_sum = tt::CB::c_intermed5;
    constexpr uint32_t cb_prev_sum = tt::CB::c_intermed6;
    constexpr uint32_t cb_exp_max_diff = tt::CB::c_intermed7;

    constexpr uint32_t cb_out = tt::CB::c_out0;


    mm_init();

    // Get Q chunk
    cb_wait_front(cb_q_in, q_chunk_tiles * q_chunks_per_core);
    cb_wait_front(cb_k_in, Sk_chunk_t * DHt * q_chunks_per_core);
    cb_wait_front(cb_v_in, DHt * Sk_chunk_t *q_chunks_per_core);
    cb_wait_front(cb_identity_scale_in, 1);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {

                uint32_t q_chunk = local_q_start + q_iter;

                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {

                    /* QK = Q_CHUNK @ K_CHUNK */
                    unpack_reconfig_data_format(cb_k_in, cb_q_in);
                    pack_reconfig_data_format(cb_qk_im);

                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w, true /*transpose*/);

                    /* QK -= cb_cur_max */
                    /* QK = exp(QK)*/
                    /* cb_cur_sum = sum(cb_qk_im, dim=-1) */
                    UNPACK(ASSERT(cb_interface[cb_identity_scale_in].fifo_rd_ptr == 19124));
                    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, Sk_chunk_t>();
                    /* OUT_IM = QK @ V_CHUNK */
                    cb_wait_front(cb_qk_im, qk_chunk_tiles);
                    unpack_reconfig_data_format(cb_v_in, cb_qk_im);
                    pack_reconfig_data_format(cb_out_im);
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w, false /*transpose*/);
                    cb_pop_front(cb_qk_im, qk_chunk_tiles);
                    cb_pop_front(cb_cur_sum, Sq_chunk_t);
                }

                cb_pop_front(cb_q_in, q_chunk_tiles);
            }
        }
    }
}
}
