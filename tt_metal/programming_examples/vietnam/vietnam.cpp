#include <iostream>
#include <memory>

#include "common/bfloat16.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt;

uint16_t round_to_nearest_even(float val) {
    uint _val = reinterpret_cast<uint &>(val);
    return static_cast<ushort>((_val + ((_val >> 16) & 1) + ((uint)0x7FFF)) >> 16);
}

float bf16_to_float(uint16_t bf16) {
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.u = bf16 << 16;
    return tmp.f;
}

int main() {
    // create device object and get command queue.
    constexpr int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue &cq = device->command_queue();

    //////////////////////////////////////////////////////////////////////////////////
    // Allocate host buffer0.
    // Fill host buffer0 with neg, zero, pos for each tile.
    //////////////////////////////////////////////////////////////////////////////////
    auto host_buffer_size = 3 * 1024 * 2;

    auto host_buffer0 = std::shared_ptr<void>(malloc(host_buffer_size), free);
    auto host_buffer0_ptr = reinterpret_cast<uint16_t *>(host_buffer0.get());
    for (int i = 0; i < 1024; ++i) {
        host_buffer0_ptr[i] = round_to_nearest_even(-i % 32);
    }
    for (int i = 0; i < 1024; ++i) {
        host_buffer0_ptr[1024 + i] = round_to_nearest_even(0);
    }
    for (int i = 0; i < 1024; ++i) {
        host_buffer0_ptr[2048 + i] = round_to_nearest_even(i % 32);
    }

    auto host_buffer1 = std::shared_ptr<void>(malloc(host_buffer_size), free);

    /////////////////////////////////////////////////////////////////////////////////
    // TODO: Allocate Device Buffer 0,1
    /////////////////////////////////////////////////////////////////////////////////
    uint32_t num_tiles = 3 /*TODO*/;
    auto page_size = 1024 * sizeof(uint16_t) /*TODO*/;
    auto device_buffer_size = num_tiles * page_size /*TODO*/;

    auto device_buffer0_config = InterleavedBufferConfig{
        .device = device,
        .size = device_buffer_size /*TODO*/,
        .page_size = page_size /*TODO*/,
        .buffer_type = BufferType::DRAM};
    auto device_buffer0 = CreateBuffer(device_buffer0_config);

    auto device_buffer1_config = InterleavedBufferConfig{
        .device = device,
        .size = device_buffer_size /*TODO*/,
        .page_size = page_size /*TODO*/,
        .buffer_type = BufferType::DRAM};
    auto device_buffer1 = CreateBuffer(device_buffer1_config);

    std::cout << "device_buffer0 total size : " << device_buffer0->size() << std::endl;
    std::cout << "device_buffer0 page size : " << device_buffer0->page_size() << std::endl;
    std::cout << "device_buffer1 total size : " << device_buffer1->size() << std::endl;
    std::cout << "device_buffer1 page size : " << device_buffer1->page_size() << std::endl;

    /////////////////////////////////////////////////////////////////////////////////
    // TODO: copy host buffer0 to device buffer0
    /////////////////////////////////////////////////////////////////////////////////

    EnqueueWriteBuffer(cq, device_buffer0 /*TODO*/, host_buffer0.get(), true /*blocking*/);

    /////////////////////////////////////////////////////////////////////////////////
    // Create program instance.
    /////////////////////////////////////////////////////////////////////////////////

    Program program = CreateProgram();
    auto core = CoreCoord{0, 0};

    /////////////////////////////////////////////////////////////////////////////////
    // TODO: allocate circular buffer 0 and 1.
    /////////////////////////////////////////////////////////////////////////////////

    auto cb_num_tiles = 2;
    auto cb0_id = CB::c_in0;
    auto cb0_data_format = tt::DataFormat::Float16_b;
    auto cb0_config =
        CircularBufferConfig(
            cb_num_tiles * page_size /*size. TODO*/, {{cb0_id /*cb_id. TODO*/, cb0_data_format /*data format. TODO*/}})
            .set_page_size(cb0_id /*cb_id. TODO*/, page_size /*page size. TODO*/);
    CreateCircularBuffer(program, core, cb0_config);

    auto cb1_id = CB::c_out0;
    auto cb1_data_format = tt::DataFormat::Float16_b;
    auto cb1_config = CircularBufferConfig(cb_num_tiles * page_size, {{cb1_id, cb1_data_format}} /*TODO*/)
                          .set_page_size(cb1_id, page_size /*TODO*/);
    CreateCircularBuffer(program, core, cb1_config);

    std::cout << "cb0 total size : " << cb0_config.total_size() << std::endl;
    std::cout << "cb1 total size : " << cb1_config.total_size() << std::endl;

    /////////////////////////////////////////////////////////////////////////////////
    // TODO: Create reader, compute and writer kernel on the program
    /////////////////////////////////////////////////////////////////////////////////
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/vietnam/kernels/compute.cpp",
        core,
        ComputeConfig{
            .compile_args = {},
            .defines = {},
        });

    const uint32_t device_buffer0_is_dram = static_cast<uint32_t>(device_buffer0_config.buffer_type == BufferType::DRAM);
    const uint32_t device_buffer1_is_dram = static_cast<uint32_t>(device_buffer1_config.buffer_type == BufferType::DRAM);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/vietnam/kernels/reader.cpp" /*reader kernel path. TODO*/,
        core,
        ReaderDataMovementConfig({device_buffer0_is_dram} /*compile args. TODO*/, {} /*defined*/));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/vietnam/kernels/writer.cpp" /*writer kernel path. TODO*/,
        core,
        WriterDataMovementConfig({device_buffer1_is_dram} /*compile args. TODO*/, {} /*defined*/));

    /////////////////////////////////////////////////////////////////////////////////
    // Reader args : device_buffer0_addr, cb0_id, num_tiles
    // Compute args : cb0_id, cb1_id, num_tiles
    // Writer args : device_buffer1_addr, cb1_id, num_tiles
    /////////////////////////////////////////////////////////////////////////////////

    const std::vector<uint32_t> reader_runtime_args = {
        device_buffer0->address(), static_cast<uint32_t>(cb0_id), num_tiles /*TODO*/};
    SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

    const std::vector<uint32_t> compute_runtime_args = {
        static_cast<uint32_t>(cb0_id), static_cast<uint32_t>(cb1_id), num_tiles /*TODO*/};
    SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

    const std::vector<uint32_t> writer_runtime_args = {
        device_buffer1->address(), static_cast<uint32_t>(cb1_id), num_tiles /*TODO*/};
    SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

    //////////////////////////////////////////////////////////////////////////////////
    // EnqueueProgram
    // Copy device buffer1 to host buffer1
    //////////////////////////////////////////////////////////////////////////////////
    EnqueueProgram(cq, program, true /*blocking*/);

    EnqueueReadBuffer(cq, device_buffer1, host_buffer1.get(), true /*blocking*/);
    auto host_buffer1_ptr = reinterpret_cast<uint16_t *>(host_buffer1.get());
    for (int tile = 0; tile < num_tiles; ++tile) {
        for (int r = 0; r < 32; ++r) {
            for (int c = 0; c < 32; ++c) {
                std::cout << bf16_to_float(host_buffer1_ptr[tile * 1024 + r * 32 + c]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n\n\n";
    }

    CloseDevice(device);

    return 0;
}
