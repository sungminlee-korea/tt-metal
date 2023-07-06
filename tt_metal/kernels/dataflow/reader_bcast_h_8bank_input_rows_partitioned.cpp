#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t src0_addr  = dataflow::get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles  = dataflow::get_arg_val<uint32_t>(3);
    uint32_t src1_addr  = dataflow::get_arg_val<uint32_t>(4);
    // skip args 1,2,5,6,7 for compat with single bank readers and reader_diff_lengths
    uint32_t NCHtWt     = dataflow::get_arg_val<uint32_t>(8);
    uint32_t NC         = dataflow::get_arg_val<uint32_t>(9);
    uint32_t Ht         = dataflow::get_arg_val<uint32_t>(10);
    uint32_t Wt         = dataflow::get_arg_val<uint32_t>(11);
    uint32_t nc1        = dataflow::get_arg_val<uint32_t>(12); // if 1 we expect the bcast tensor to have NC=1
    uint32_t start_id   = dataflow::get_arg_val<uint32_t>(13);
    uint32_t HtWt       = dataflow::get_arg_val<uint32_t>(14); // HtWt of input tensor

    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool src0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    uint32_t tile_bytes = dataflow::get_tile_size(cb_id_in0);

    const dataflow::InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    const dataflow::InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i = 0;
    uint32_t i1 = 0;
    uint32_t i_nc = 0;
    for (uint32_t nc = 0; nc < NC; nc++) {
        i = i_nc + start_id;
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                dataflow::cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = dataflow::get_write_ptr(cb_id_in0);
                dataflow::noc_async_read_tile(i, s0, l1_write_addr_in0);
                dataflow::noc_async_read_barrier();
                dataflow::cb_push_back(cb_id_in0, onetile);

                // for each W-tile of the first tensor we push one tile from the second arg tile list
                // but we loop the second list around
                dataflow::cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = dataflow::get_write_ptr(cb_id_in1);
                dataflow::noc_async_read_tile(i1, s1, l1_write_addr_in1);
                dataflow::noc_async_read_barrier();
                dataflow::cb_push_back(cb_id_in1, onetile);
                i1 ++;
                i ++; // input tile iterates over NC Ht Wt
            }

            // bcast tensor should be NC1W (actually NC32W padded with 0s in H)
            // wrap W around for each h (broadcast)
            i1 -= Wt;
        }
        // we reused Wt tiles out of NCWt bcast tensor Ht times, now advance for next NC
        if (nc1 == 0) // if bcast NC==1 we don't advance but reuse the tensor
            i1 += Wt;
        i_nc += HtWt;
    }
}
