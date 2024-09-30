// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_softmax_w_large(const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op, const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    log_info(LogTest, "Large tensor algorithm selected");
    // split work
    auto shape = input.get_legacy_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    auto num = input.volume() / H / W;

    uint32_t num_kernel_rows = num * Ht;
    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, num_kernel_rows);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] = get_compute_kernel_config_args(arch, compute_kernel_config);

    Program program = Program();

    // create circular buffers
    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::cb_0, 2},         // input
            {CB::cb_1, 1},         // mask
            {CB::cb_2, 1},          // scaler
            {CB::cb_16, 2},        // output
            {CB::cb_24, 2, intermed_data_format},   // exp(x)
            {CB::cb_25, 1, intermed_data_format},   // reduce
            {CB::cb_26, 1, intermed_data_format},   // syn
            {CB::cb_27, 1, intermed_data_format},   // max
            {CB::cb_28, 1, intermed_data_format},   // tmp
        });

    // create read/wrtie kernel
    bool src_is_dram = input.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax/kernels/reader_moreh_softmax_w_large.cpp", all_cores, {src_is_dram}, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax/kernels/writer_moreh_softmax_w_large.cpp", all_cores, {dst_is_dram}, writer_defines);

    std::map<string, string> compute_defines;
    if (op == MorehSoftmaxOp::SOFTMAX || op == MorehSoftmaxOp::LOGSOFTMAX) compute_defines["SOFTMAX"] = "1";
    else compute_defines["SOFTMIN"] = "1";

    if (op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["LOG"] = "1";
    }

    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // create compute kernel
    CreateComputeKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_softmax/kernels/moreh_softmax_w_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Wt}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Wt}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        float scaler = 1.0f;
        uint32_t mask_w = shape.without_padding()[-1] % TILE_WIDTH;
        if(mask_w == 0) mask_w = TILE_WIDTH;
        vector<uint32_t> reader_args = {
            input.buffer()->address(), num_tiles_per_core, tile_offset, Wt, *reinterpret_cast<uint32_t *>(&scaler), mask_w};

        vector<uint32_t> writer_args = {output.buffer()->address(), num_tiles_per_core, tile_offset, Wt};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core * Wt;
    }

    return {
        .program = std::move(program),
        .override_runtime_arguments_callback =
            create_override_runtime_arguments_callback(reader_kernel_id, writer_kernel_id, num_cores, core_h)};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
