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

//#define DEBUG 1

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
    MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(in0_cb, scale_cb) ));

    const uint32_t num_tiles = rows * cols;
    //cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst(tt::DstMode::Half);
        for (uint32_t j = 0; j < cols; j++) {

            // MATH(( llk_math_reduce<pool_type, reduce_dim, MATH_FIDELITY, DST_ACCUM_MODE>(0) ));
            MATH(( llk_math_eltwise_binary<ELWMUL, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(0, 0, 0) ));
            // MATH(( llk_math_reduce<pool_type, reduce_dim, MATH_FIDELITY, DST_ACCUM_MODE>(0) ));
            UNPACK(( llk_unpack_AB(in0_cb, scale_cb, 0, 0) ));
            // reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, 0, 0, 0);

            // MATH(( llk_math_eltwise_unary_datacopy<A2D>(0, scale_cb) ));
            // UNPACK(( llk_unpack_A(scale_cb, 0) ));
        }

        cb_reserve_back(out_cb, 1);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
        //print_full_tile(out_cb,i);
    }
    // tensix_sync();

   reduce_revert_delta<reduce_dim>(out_cb);
//    tensix_sync(); UNCOMMENT FOR DETERMINISM
}

inline void add_nops(const int num_nops) {
	for(int i = 0; i < num_nops; i++) {
		TTI_NOP;
        }
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
    #ifdef MM_ADD_NOPS
    UNPACK(DPRINT<<MM_NUM_NOPS << "\n");
    UNPACK(add_nops(MM_NUM_NOPS));
    #endif

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


    add_nops(1000);
    mm_init();
    cb_wait_front(cb_q_in, q_chunk_tiles * q_chunks_per_core);
    cb_wait_front(cb_k_in, Sk_chunk_t * DHt * q_chunks_per_core);
    cb_wait_front(cb_v_in, DHt * Sk_chunk_t *q_chunks_per_core);
    cb_wait_front(cb_mask_in, qk_chunk_tiles * q_chunks_per_core);
    cb_wait_front(cb_identity_scale_in, 1);
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk = local_q_start + q_iter;

                // Get Q chunk
                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;
                // cb_wait_front(cb_q_in, q_chunk_tiles);

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    /* QK = Q_CHUNK @ K_CHUNK */
                    unpack_reconfig_data_format(cb_k_in, cb_q_in);
                    pack_reconfig_data_format(cb_qk_im);
                    // tensix_sync();
                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w, true /*transpose*/);

                    /* QK *= SCALE */
                    // tensix_sync();
                    // mul_block_bcast_scalar_inplace<cb_qk_im, cb_scale_in, qk_chunk_tiles>();

                    // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                    // Q-range = [q_low, q_high)
                    // K-range = [k_low, k_high)
                    // does_overlap = not (q_low >= k_high or k_low >= q_high)
                    // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                    if (!(q_low_idx >= k_high_idx)) {
                        /* QK += MASK */
                        // unpack_reconfig_data_format(cb_qk_im, cb_mask_in);
                        // tensix_sync();
                        // add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                        // cb_wait_front(cb_mask_in, qk_chunk_tiles);
                        cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    // unpack_reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                    // tensix_sync();
                    // reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, cb_cur_max, Sq_chunk_t, Sk_chunk_t>();

                    if (k_chunk > 0) {
                        // tensix_sync();
                        // max_block_inplace<cb_cur_max, cb_prev_max, Sq_chunk_t>();
                    }

                    /* QK -= cb_cur_max */
                    /* QK = exp(QK)*/
                    // tensix_sync();
                    // sub_exp_block_bcast_cols_inplace<cb_qk_im, cb_cur_max, Sq_chunk_t, Sk_chunk_t>();

                    /* cb_cur_sum = sum(cb_qk_im, dim=-1) */
                    // tensix_sync();
                    // PACK(DPRINT << "cb_identity_scale_in " << ENDL() );
                    // print_cb_details(cb_identity_scale_in);
                    UNPACK(ASSERT(cb_interface[cb_identity_scale_in].fifo_rd_ptr == 19124));
                    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, Sk_chunk_t>();

                    // MATH(TTI_ZEROACC(p_zeroacc::CLR_ALL, ADDR_MOD_0, 0);)
                    // MATH(tensix_sync();)

                    // UNPACK(TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC_RESET_ALL_BANKS);)
                    // UNPACK(TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC_RESET_ALL_BANKS);)

                    /* OUT_IM = QK @ V_CHUNK */

                    // UNPACK(tensix_sync());
                    // PACK(DPRINT << "cb_qk_im " << ENDL() );
                    // print_cb_details(cb_qk_im);
                    cb_wait_front(cb_qk_im, qk_chunk_tiles);
                    unpack_reconfig_data_format(cb_v_in, cb_qk_im);
                    pack_reconfig_data_format(cb_out_im);
                    // volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(23604);
                    UNPACK(ASSERT(cb_interface[cb_qk_im].fifo_rd_ptr == 19252));
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w, false /*transpose*/);
                    volatile tt_l1_ptr uint32_t* in1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(205632);
                    volatile tt_l1_ptr uint32_t* in1_second_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(238400);
                    volatile tt_l1_ptr uint32_t* in_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(308032);
                    volatile tt_l1_ptr uint32_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(377664);
                    PACK(ASSERT(in1_ptr[0] != 0x7F800000));
                    PACK(ASSERT(in1_second_ptr[0] != 0x7F800000));
                    PACK(ASSERT(in_ptr[0] != 0x7F800000));
                    // PACK(ASSERT(out_ptr[0] != 0x7F800000));
                    // PACK(DPRINT << "cb_qk_im");
                    // print_full_tile(cb_qk_im);
                    // PACK(DPRINT << "cb_v_in");
                    // print_full_tile(cb_v_in);
                    // print_cb_details(cb_out);
                    // PACK(DPRINT << "cb_out");
                    // print_full_tile(cb_out);

                    cb_pop_front(cb_qk_im, qk_chunk_tiles);
                    // tensix_sync();

                    /* OUT_ACC += OUT_IM */
                    if (k_chunk == 0) {
                        // tensix_sync();
                        // unpack_reconfig_data_format_srca(cb_out_im);
                        // pack_reconfig_data_format(cb_out_accumulate_im);
                        // copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                    } else {
                        // tensix_sync();
                        /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
                        // sub_exp_block(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                        // cb_pop_front(cb_prev_max, Sq_chunk_t);

                        /* cb_prev_sum *= cb_exp_max_diff */
                        // tensix_sync();
                        // mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                        /* cb_out_accumulate_im *= cb_exp_max_diff */
                        // tensix_sync();
                        // mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, DHt);

                        /* cb_cur_sum += cb_prev_sum */
                        // tensix_sync();
                        // add_block_inplace(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                        /* cb_out_accumulate_im += cb_out_im */
                        // tensix_sync();
                        // add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                    }

                    // Set cb_prev_sum and cb_prev_max
                    // tensix_sync();
                    // copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    // tensix_sync();
                    // copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                    cb_pop_front(cb_cur_sum, Sq_chunk_t);
                }

                /* cb_cur_sum = 1.0 / cb_cur_sum */
                // cb_push_back(cb_cur_sum, Sq_chunk_t);
                // tensix_sync();
                // recip_block_inplace(cb_cur_sum, Sq_chunk_t);

                /* cb_out_accumulate_im *= cb_cur_sum */
                // tensix_sync();
                // mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, DHt);
                // unpack_reconfig_data_format_srca(cb_out_accumulate_im);
                // pack_reconfig_data_format(cb_out);
                // // // tensix_sync();
                // copy_block(cb_out_accumulate_im, cb_out, out_chunk_tiles);

                cb_pop_front(cb_q_in, q_chunk_tiles);
                // free up cb_prev_max after K chunks
                // cb_pop_front(cb_prev_max, Sq_chunk_t);
                // cb_pop_front(cb_prev_sum, Sq_chunk_t);
            }
        }
    }


}
}
