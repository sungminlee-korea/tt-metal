// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step2_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {
std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative2(float ord) {
    auto p = std::floor(ord);
    auto decimal = ord - p;
    const bool p_is_negative = p < 0.0f;
    if (p_is_negative) {
        p = -p;
    }
    return std::make_tuple(static_cast<uint32_t>(p), decimal, p_is_negative);
}

MorehClipGradNormStep2Operation::ProgramFactory::cached_program_t
MorehClipGradNormStep2Operation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto norm_type = operation_attributes.norm_type;
    const auto& tmp_pow_sum = tensor_args.tmp_pow_sum;
    const auto& total_norm = tensor_args.total_norm;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = tmp_pow_sum.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_tiles = tmp_pow_sum.volume() / tt::constants::TILE_HW;

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative2(1.0f / norm_type);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord single_core = {0, 0};

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(==tmp_pow_sum)
    const uint32_t in1_t = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    const uint32_t out0_t = 1;  // output(==total_norm)

    const uint32_t im0_t = 1;  // Sum[tmp_pow_sum](==x)
    const uint32_t im1_t = 1;  // x^p
    const uint32_t im2_t = 1;  // log(x)
    const uint32_t im3_t = 1;  // exp(log(x) * decimal)

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(total_norm.get_dtype());

    tt::operations::primary::CreateCircularBuffer(
        program,
        single_core,
        cb_data_format,
        {
            {tt::CB::c_in0, in0_t},        // input(==tmp_pow_sum)
            {tt::CB::c_in1, in1_t},        // decimal
            {tt::CB::c_out0, out0_t},      // output(==total_norm)
            {tt::CB::c_intermed0, im0_t},  // Sum[tmp_pow_sum](==x)
            {tt::CB::c_intermed1, im1_t},  // x^p
            {tt::CB::c_intermed2, im2_t},  // log(x)
            {tt::CB::c_intermed3, im3_t},  // exp(log(x) * decimal)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/kernels/"
        "reader_moreh_clip_grad_norm_step2.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/kernels/"
        "writer_moreh_clip_grad_norm_step2.cpp";

    const auto reader_kernels_id = tt::operations::primary::CreateReadKernel(program, reader_kernel_file, single_core);
    const auto writer_kernels_id = tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, single_core);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/kernels/"
        "moreh_clip_grad_norm_step2_kernel.cpp";

    const auto compute_kernels_id =
        tt::operations::primary::CreateComputeKernel(program, compute_kernel_file, {single_core, num_tiles});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = tmp_pow_sum.buffer()->address();
    const auto output_addr = total_norm.buffer()->address();

    // reader
    const std::vector<uint32_t> reader_runtime_args{
        input_addr,
        static_cast<uint32_t>(tt::operations::primary::is_dram(tmp_pow_sum)),
        num_tiles,
        *reinterpret_cast<uint32_t*>(&decimal)};
    SetRuntimeArgs(program, reader_kernels_id, single_core, reader_runtime_args);

    // writer
    const std::vector<uint32_t> writer_runtime_args{
        output_addr, static_cast<uint32_t>(tt::operations::primary::is_dram(total_norm))};
    SetRuntimeArgs(program, writer_kernels_id, single_core, writer_runtime_args);

    // compute
    const std::vector<uint32_t> compute_runtime_args{num_tiles, p, static_cast<uint32_t>(p_is_negative)};
    SetRuntimeArgs(program, compute_kernels_id, single_core, compute_runtime_args);

    return {std::move(program), {reader_kernels_id, writer_kernels_id, compute_kernels_id, single_core}};
}

void MorehClipGradNormStep2Operation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto norm_type = operation_attributes.norm_type;
    const auto& tmp_pow_sum = tensor_args.tmp_pow_sum;
    const auto& total_norm = tensor_args.total_norm;

    auto& program = cached_program.program;
    auto& reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto& writer_kernels_id = cached_program.shared_variables.writer_kernels_id;
    auto& compute_kernels_id = cached_program.shared_variables.compute_kernels_id;
    auto& single_core = cached_program.shared_variables.single_core;

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative2(1.0f / norm_type);

    const auto input_address = tmp_pow_sum.buffer()->address();
    const auto output_address = total_norm.buffer()->address();

    {
        auto& runtime_args = GetRuntimeArgs(program, reader_kernels_id, single_core);
        runtime_args[0] = input_address;
        runtime_args[3] = *reinterpret_cast<uint32_t*>(&decimal);
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, writer_kernels_id, single_core);
        runtime_args[0] = output_address;
    }

    {
        auto& runtime_args = GetRuntimeArgs(program, compute_kernels_id, single_core);
        runtime_args[1] = p;
        runtime_args[2] = static_cast<uint32_t>(p_is_negative);
    }
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
