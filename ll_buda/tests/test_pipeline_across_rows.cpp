#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        std::vector<tt_xy_pair> cores = {{0,0}, {1,0}};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src_dram_buffer = ll_buda::CreateDramBuffer(0, dram_buffer_size, 0);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(7, dram_buffer_size, 0);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates(device);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates(device);

        uint32_t cb_index = 8;
        uint32_t cb_addr = 250 * 1024;
        uint32_t cb_size_tiles = 16;
        auto cb1 = ll_buda::CreateCircularBuffer(
            program,
            cb_index,
            cores[0],
            cb_size_tiles,
            cb_size_tiles * single_tile_size,
            cb_addr,
            tt::DataFormat::Float16_b
        );
        
        auto reader_first_stage_kernel_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[0], {8, 2});
        auto writer_last_stage_kernel_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[0], {8, 4});

        auto reader_first_stage_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_first_stage.cpp",
            cores[0],
            reader_first_stage_kernel_args,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto writer_last_stage_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_last_stage.cpp",
            cores[0],
            writer_last_stage_kernel_args,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(device, src_dram_buffer, src_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            reader_first_stage_kernel,
            cores[0],
            {src_dram_buffer->address(),
            (uint32_t)dram_src_noc_xy.x,
            (uint32_t)dram_src_noc_xy.y,
            (uint32_t)num_tiles});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            writer_last_stage_kernel,
            cores[0],
            {dst_dram_buffer->address(),
            (uint32_t)dram_dst_noc_xy.x,
            (uint32_t)dram_dst_noc_xy.y,
            (uint32_t)num_tiles});

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
