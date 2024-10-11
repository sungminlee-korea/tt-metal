// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "utils/bfloat16.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t core_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_intermed0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_intermed1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(3);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr uint32_t in0_stick_size = get_compile_time_arg_val(6);
    constexpr uint32_t intermed0_stick_size = get_compile_time_arg_val(7);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(8);
    constexpr uint32_t B = get_compile_time_arg_val(9);
    constexpr uint32_t C = get_compile_time_arg_val(10);
    constexpr uint32_t H = get_compile_time_arg_val(11);
    constexpr uint32_t W = get_compile_time_arg_val(12);
    constexpr uint32_t num_cores = get_compile_time_arg_val(13);
    uint32_t semaphore_addr_ptr = get_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t final_cores_physical_x = get_compile_time_arg_val(15);
    constexpr uint32_t final_cores_physical_y = get_compile_time_arg_val(16);
    bool reducer_core = core_id == 0? 1 : 0;

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = in0_stick_size};
    uint64_t src_noc_addr = get_noc_addr(0, s0);
    // Use cb as L1 scratch memory
    uint32_t intermed0_addr = get_write_ptr(cb_id_intermed0);
    volatile tt_l1_ptr uint32_t* max_ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(intermed0_addr);

    uint32_t intermed1_addr = get_write_ptr(cb_id_intermed1) + 4*num_cores;
    volatile tt_l1_ptr uint16_t* max_vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(intermed1_addr);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);
    //DPRINT << cb_id_intermed0 << " "<< cb_id_intermed1 << " " <<intermed0_addr << " " << intermed1_addr <<" " << cb_addr<< ENDL();

    uint32_t max_index = 0;
    uint32_t max_val = 0;
    uint32_t index_counter = 0;
    uint32_t core_offset = core_id*W;
    // uint32_t page_number = core_id; //(core_id*W)/in0_stick_size;
    // uint32_t page_offset = 0; //(core_id*W)%in0_stick_size;
    // DPRINT << "core" << core_id << " page_number: " << page_number<< " page_offset: " << page_offset << "page size" <<  in0_stick_size <<ENDL();
    noc_async_read(src_noc_addr, cb_addr, (W*num_cores)*2);
    //noc_async_read(src_noc_addr + core_id*W*2, cb_addr, (W)*2);
    //noc_async_read_page(0, s0, cb_addr);
    noc_async_read_barrier();

    index_counter = core_offset;
    max_index = index_counter;
    max_val = stick[0]; //
    max_val = stick[core_id*W];
    for(uint32_t i = core_id*W; i < W + core_id*W; i++) {
        uint16_t val = stick[i];
        //DPRINT << "W"<< i<< ":"<<val << ENDL();
        if(bfloat16_greater(val, max_val)) {
            max_index = index_counter;
            max_val = val;
        }
        index_counter++;

    }

    // set max_vals for reader and writer kernels
    max_ids[core_id] = max_index;
    max_vals[core_id] = max_val;

    // DPRINT << "core" << core_id << " max_index: " << max_ids[0]<< " max_val: " << max_vals[0] << ENDL();
    // DPRINT << "core" << core_id << max_index << " done" << max_val<< ENDL();
    // write max_vals to reducer core CB
    uint64_t dst_cb_addr_0 = get_noc_addr(final_cores_physical_x, final_cores_physical_y, get_write_ptr(cb_id_intermed0));
    uint64_t dst_cb_addr_1 = get_noc_addr(final_cores_physical_x, final_cores_physical_y, get_write_ptr(cb_id_intermed1))+4*num_cores;
    //noc_async_write(intermed_addr + (core_id%2)*4, dst_cb_addr + core_id*4, 4);
    noc_async_write(intermed0_addr+ core_id*4, dst_cb_addr_0 + core_id*4, 4);
    noc_async_write_barrier();
    noc_async_write(intermed1_addr+ core_id*2, dst_cb_addr_1 + core_id*2, 2);
    noc_async_write_barrier();


    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr_0);
    volatile tt_l1_ptr uint16_t* ptr1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_cb_addr_1);
    // DPRINT << dst_cb_addr_0 << " "<< dst_cb_addr_0 + core_id*4 <<" "<< dst_cb_addr_1 << " " <<intermed0_addr << " " << intermed1_addr <<" " << cb_addr<< ENDL();
    // DPRINT << ptr[0] <<" "<< ptr[1] << " " <<ptr[2] << " " << ptr[3] << " " << ptr[4]<<ENDL();
    // DPRINT << ptr1[0] <<" "<< ptr1[1] << " " <<ptr1[2] << " " << ptr1[3] << " " << ptr1[4]<<ENDL();

    // inc noc semaphore
    const uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(final_cores_physical_x, final_cores_physical_y, semaphore_addr_ptr);
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    //DPRINT << "core" << core_id << " done" << ENDL();

    if (reducer_core) {
        // wait for semaphore
        volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr_ptr);
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores);
        // Use cb as L1 scratch memory
        uint32_t out_addr = get_write_ptr(cb_id_out0);
        volatile tt_l1_ptr uint32_t* max_vals_final = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
        // re-use intermed cb
        uint32_t intermed0_re_addr = get_write_ptr(cb_id_intermed0);
        volatile tt_l1_ptr uint32_t* max_ids_reduce = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(intermed0_re_addr);
        uint32_t intermed1_re_addr = get_write_ptr(cb_id_intermed1) + 4*num_cores;
        volatile tt_l1_ptr uint16_t* max_vals_reduce = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(intermed1_re_addr);

        max_index = max_ids_reduce[0];
        max_val = max_vals_reduce[0];
        for(uint32_t i = 0; i < num_cores; i++) {
            uint32_t index = max_ids_reduce[i];
            uint16_t val = max_vals_reduce[i];
            //DPRINT << "core"<< i<< ":"<<index <<" "<< val<< ENDL();
            if(bfloat16_greater(val, max_val)) {
                max_index = index;
                max_val = val;
            }
        }
        max_vals_final[0] = max_index;

        const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};
        uint64_t dst_noc_addr = get_noc_addr(0, s_out);
        noc_async_write(out_addr, dst_noc_addr, out_stick_size);
        noc_async_write_barrier();
    }
}
