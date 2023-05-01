#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <iomanip>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"
// #include "tt_gdb/tt_gdb.hpp"

#include "llrt.hpp"
#include "common/bfloat16.hpp"
#include "test_libs/tiles.hpp"

#include "tt_metal/tools/profiler/profiler.hpp"

uint32_t NUM_TILES = 400;

using tt::llrt::CircularBufferConfigVec;

bool run_data_copy_multi_tile(tt_cluster* cluster, int chip_id, const vector<tt_xy_pair> cores, int num_tiles) {

    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    std::uint32_t host_buffer_src_addr = 0;
    std::uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_dst_channel_id = 0;
    log_info(tt::LogVerif, "num_tiles = {}", num_tiles);
    log_info(tt::LogVerif, "single_tile_size = {} B", single_tile_size);
    log_info(tt::LogVerif, "dram_bufer_size = {} B", dram_buffer_size);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, cores);

    tt_xy_pair pcie_core_coordinates = {0, 4};

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, NUM_TILES*single_tile_size, NUM_TILES);
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 990*1024, 1*single_tile_size, 1);

    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, cores.at(0), circular_buffer_config_vec);
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, cores.at(1), circular_buffer_config_vec);

    tt_xy_pair dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);


    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, cores.at(0),
        { host_buffer_src_addr, (std::uint32_t)pcie_core_coordinates.x, (std::uint32_t)pcie_core_coordinates.y, (std::uint32_t)num_tiles },
        NCRISC_L1_ARG_BASE);


    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, cores.at(1),
        { host_buffer_src_addr, (std::uint32_t)pcie_core_coordinates.x, (std::uint32_t)pcie_core_coordinates.y, (std::uint32_t)num_tiles },
        BRISC_L1_ARG_BASE);

    // Note: TRISC 0/1/2 kernel args are hard-coded

    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    // Write tiles sequentially to DRAM
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, tt::tiles_test::get_seed_from_systime());

    // Instead of writing DRAM vec, we write to host memory and have the device pull from host
    cluster->write_sysmem_vec(src_vec, 0, 0);

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, cores);
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, cores);

    static Profiler p = Profiler();
    p.dumpDeviceResults(cluster, 0, cores);

    // std::vector<std::uint32_t> dst_vec;
    // cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);

    // bool pass = (dst_vec == src_vec);

    return true;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        string op = "two_readers";
        string op_path = "built_kernels/" + op;

        int chip_id = 0;
        const vector<tt_xy_pair> cores = {{1, 11}, {6, 11}};

        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", chip_id, cores.at(0), 1); // ncrisc
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", chip_id, cores.at(1), 0); // brisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, cores.at(0), 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, cores.at(0), 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, cores.at(0), 2); // trisc2
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, cores.at(1), 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, cores.at(1), 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, cores.at(1), 2); // trisc2

        if (pass) {
            const vector<string> ops = {op};

            // tt_gdb::tt_gdb(cluster, chip_id, cores, ops);
            pass &= run_data_copy_multi_tile(cluster, chip_id, cores, NUM_TILES); // must match the value in test_compile_datacopy!
        }

        cluster->close_device();
        delete cluster;

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
