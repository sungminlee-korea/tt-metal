// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_test2/moreh_test2_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_test2_impl(
    const Tensor &input,
    const Tensor &input2,
    const Tensor &output,
    float p,
    const DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device{input.device()};
    auto program{CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_tiles{input.volume() / tt::constants::TILE_HW};
    tt::DataFormat data_format{tt_metal::datatype_to_dataformat_converter(input.get_dtype())};
    tt::DataFormat input2_data_format{tt_metal::datatype_to_dataformat_converter(input2.get_dtype())};

    log_debug(LogOp, "input data format {}", data_format);
    log_debug(LogOp, "input2 data format {}", input2_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // Use single core for testing
    // auto grid{device->compute_with_storage_grid_size()};
    CoreCoord grid{1, 1};
    const auto num_cores_y{grid.y};

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_tiles);

    log_debug(LogOp, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogOp, "num_tiles_per_core_group_1 : {}", num_tiles_per_core_group_1);
    log_debug(LogOp, "num_tiles_per_core_group_2 : {}", num_tiles_per_core_group_2);
    log_debug(LogOp, "num_tiles {}", num_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{2};        // input, double buffer
    const uint32_t in1_t{2};        // input2, double buffer
    const uint32_t in2_t{1};        // temp
    const uint32_t intermed0_t{1};  // temp
    const uint32_t intermed1_t{1};  // temp
    const uint32_t out0_t{2};       // output, double buffer

    log_debug(LogOp, "tile size for CB in0_t {}", in0_t);
    log_debug(LogOp, "tile size for CB in1_t {}", in1_t);
    log_debug(LogOp, "tile size for CB in2_t {}", in2_t);
    log_debug(LogOp, "tile size for CB intermed0_t {}", intermed0_t);
    log_debug(LogOp, "tile size for CB intermed1_t {}", intermed1_t);
    log_debug(LogOp, "tile size for CB out0_t {}", out0_t);

    // input, temp, output
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {{CB::c_in0, in0_t},
         {CB::c_in2, in2_t},
         {CB::c_out0, out0_t},
         {CB::c_intermed0, intermed0_t},
         {CB::c_intermed1, intermed1_t}});

    // input2
    CreateCircularBuffer(
        program,
        all_cores,
        input2_data_format,
        {
            {CB::c_in1, in1_t}  // input2
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // 1 / (1 - p)
    float p_complement_inverse{(1.0f / (1.0f - p))};
    uint32_t pci_uint32_data = *reinterpret_cast<uint32_t *>(&p_complement_inverse);
    log_debug(LogOp, "p {} p_complement_inverse {}", p, p_complement_inverse);

    // Wt
    uint32_t input_w = input.get_legacy_shape()[-1];
    uint32_t input_Wt = (input_w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(is_dram(input)), static_cast<uint32_t>(is_dram(input2)), pci_uint32_data, input_Wt};
    const std::string reader_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/kernels/reader_test2.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(output))};
    const std::string writer_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/kernels/writer_test2.cpp";
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{input_Wt};

    std::map<string, string> compute_defines;
    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/kernels/moreh_test2.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_tiles_per_core_group_1, compute_args_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{input_Wt};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_tiles_per_core_group_2, compute_args_group_2},
            compute_defines,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode);
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // input, input2, output
    const auto input_addr{input.buffer()->address()};
    const auto input2_addr{input2.buffer()->address()};
    const auto output_addr{output.buffer()->address()};
    log_debug(LogOp, "input_addr {}", input_addr);
    log_debug(LogOp, "input2_addr {}", input2_addr);
    log_debug(LogOp, "output_addr {}", output_addr);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_to_be_used; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_tiles_per_core_group_1;
            SetRuntimeArgs(program, compute_kernel_1_id, core, {num_output_tiles_per_core, num_tiles_written});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_tiles_per_core_group_2;
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_output_tiles_per_core, num_tiles_written});
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {input_addr, input2_addr, num_output_tiles_per_core, num_tiles_written});

        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output_addr, num_output_tiles_per_core, num_tiles_written});

        num_tiles_written += num_output_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_debug(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto *input_buffer = input_tensors.at(0).buffer();
        const auto *input2_buffer = input_tensors.at(1).buffer();
        const auto *output_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_buffer->address();
                runtime_args[1] = input2_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
