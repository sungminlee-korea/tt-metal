// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_norm_other_impl(const Tensor &input, float p, int64_t dim, const Tensor &output, const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using namespace tt::constants;
   ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_legacy_shape();
    const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

    const auto H = input_shape[-2];
    const auto W = input_shape[-1];

    const auto Ht = H / TILE_HEIGHT;
    const auto Wt = W / TILE_WIDTH;

    const auto num_reduced_tiles_along_dim = input_shape[dim];
    const auto num_output_tiles = output.volume() / TILE_HW;

    uint32_t outer_stride{1};
    for (int64_t j = dim; j < input_rank; ++j) {
        outer_stride *= input_shape[j];
    }
    outer_stride /= TILE_HW;

    uint32_t num_inner_tiles{1};
    for (int64_t j = dim + 1; j < input_rank; ++j) {
        num_inner_tiles *= input_shape[j];
    }
    num_inner_tiles /= TILE_HW;

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_recip_p, recip_p_decimal, recip_p_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(1.0f / p);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one
    const uint32_t in2_t{1};  // decimal
    const uint32_t in3_t{1};  // recip_p_decimal

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // |x|
    const uint32_t im1_t{1};  // log(|x|)
    const uint32_t im2_t{1};  // exp(log(|x|) * decimal)
    const uint32_t im3_t{1};  // |x|^p
    const uint32_t im4_t{1};  // |x|^p * exp(log(|x|) * decimal) == |x + decimal|^p
    const uint32_t im5_t{1};  // Add(|x + decimal|^p)

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::cb_0, in0_t},    // input
            {CB::cb_1, in1_t},    // one
            {CB::cb_2, in2_t},    // decimal
            {CB::cb_3, in3_t},    // recip_p_decimal
            {CB::cb_16, out0_t},  // output
            {CB::cb_24, im0_t, intermed_data_format},
            {CB::cb_25, im1_t, intermed_data_format},
            {CB::cb_26, im2_t, intermed_data_format},
            {CB::cb_27, im3_t, intermed_data_format},
            {CB::cb_28, im4_t, intermed_data_format},
            {CB::cb_29, im5_t, intermed_data_format},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_norm/moreh_norm_other/kernels/"
        "reader_moreh_norm_other.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_norm/moreh_norm_other/kernels/"
        "writer_moreh_norm_other.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_norm/moreh_norm_other/kernels/"
        "moreh_norm_other_kernel.cpp";

    const auto compute_kernels_id_1 =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_units_per_core_group_1},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        compute_kernels_id_2 =
            CreateComputeKernel(program, compute_kernel_file, {core_group_2, num_units_per_core_group_2},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core;
        KernelHandle compute_kernel_id;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_units_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_units_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            *reinterpret_cast<uint32_t *>(&decimal),
            *reinterpret_cast<uint32_t *>(&recip_p_decimal),
            num_output_tiles_per_core,
            tile_offset,
            outer_stride,
            num_inner_tiles,
            num_reduced_tiles_along_dim};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr, static_cast<uint32_t>(is_dram(output)), num_output_tiles_per_core, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_output_tiles_per_core,
            num_reduced_tiles_along_dim,
            floored_p,
            static_cast<uint32_t>(p_is_negative),
            floored_recip_p,
            static_cast<uint32_t>(recip_p_is_negative)};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_output_tiles_per_core;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernels_id, writer_kernels_id, num_cores_to_be_used, num_cores_y)};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
