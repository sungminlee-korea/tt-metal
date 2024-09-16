// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step3_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

MorehClipGradNormStep3Operation::ProgramFactory::cached_program_t
MorehClipGradNormStep3Operation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& inputs = tensor_args.inputs;
    const auto& clip_coef_clamped = tensor_args.clip_coef_clamped;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = inputs.at(0).device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_inputs_per_core_group_1,
         num_inputs_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_inputs);
    TT_ASSERT(core_group_2.ranges().empty());
    TT_ASSERT(num_inputs_per_core_group_1 == 1);
    TT_ASSERT(num_inputs_per_core_group_2 == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(inplace)
    const uint32_t in1_t = 1;  // clip_coef_clamped

    const uint32_t out0_t = 1;  // output(inplace)

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(inputs.at(0).get_dtype());

    tt::operations::primary::CreateCircularBuffer(
        program,
        core_group_1,
        cb_data_format,
        {
            {tt::CB::c_in0, in0_t},    // input(inplace)
            {tt::CB::c_in1, in1_t},    // clip_coef_clamped
            {tt::CB::c_out0, out0_t},  // output(inplace)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/kernels/"
        "reader_moreh_clip_grad_norm_step3.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/kernels/"
        "writer_moreh_clip_grad_norm_step3.cpp";

    const auto reader_kernels_id = tt::operations::primary::CreateReadKernel(program, reader_kernel_file, core_group_1);
    const auto writer_kernels_id =
        tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, core_group_1);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/kernels/"
        "moreh_clip_grad_norm_step3_kernel.cpp";

    const auto compute_kernels_id = tt::operations::primary::CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_inputs_per_core_group_1});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto clip_coef_clamped_addr = clip_coef_clamped.buffer()->address();
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = input.volume() / tt::constants::TILE_HW;

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(tt::operations::primary::is_dram(input)),
            clip_coef_clamped_addr,
            static_cast<uint32_t>(tt::operations::primary::is_dram(clip_coef_clamped)),
            num_tiles};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            input_addr, static_cast<uint32_t>(tt::operations::primary::is_dram(input)), num_tiles};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{num_tiles};
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_addresses_callback =
        [reader_kernels_id = reader_kernels_id,
         writer_kernels_id = writer_kernels_id,
         num_cores_to_be_used = num_cores_to_be_used,
         num_cores_y = num_cores_y](
            const Program& program, const std::vector<Buffer*>& input_buffers, const std::vector<Buffer*>&) {};

    return {std::move(program), {reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y}};
}

void MorehClipGradNormStep3Operation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& inputs = tensor_args.inputs;
    const auto& clip_coef_clamped = tensor_args.clip_coef_clamped;

    auto& program = cached_program.program;
    auto& reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto& writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto& num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto clip_coef_clamped_buffer = clip_coef_clamped.buffer();
    const auto clip_coef_clamped_address = clip_coef_clamped_buffer->address();

    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
            runtime_args[0] = inputs.at(i).buffer()->address();
            runtime_args[2] = clip_coef_clamped_address;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
            runtime_args[0] = inputs.at(i).buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
