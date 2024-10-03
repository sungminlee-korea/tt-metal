// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_device_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
#include "host_api.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::detail;

union datatype {
    uint32_t u32;
    float f32;
} u;

namespace ttnn::operations::index_fill {
IndexFillOperation::MultiCore::cached_program_t IndexFillOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {

    const Tensor &index = tensor_args.index;
    const Tensor &input = tensor_args.input;

    const auto input_shape = input.get_legacy_shape();
    const auto n = input_shape.rank();

    uint32_t dim = operation_attributes.dim;
    std::vector<uint32_t> strides(n);
    strides[n - 1] = 1; // Last dimension stride is 1
    for (int i = n - 2;i >= 0; i--) {
        strides[i] = strides[i + 1] * input_shape[i + 1];
    }


    auto fill_value = operation_attributes.value;
    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        u.f32 = std::get<float>(fill_value);
    }

    auto num_tiles = input.volume() / TILE_HW;
    Program program{};
    Device *device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    auto input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    auto index_data_format = datatype_to_dataformat_converter(index.get_dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size = TileSize(input_data_format);

    uint32_t input_unit_size = input.get_legacy_shape()[-1] * input.element_size();
    uint32_t rounded_input_unit_size = round_up_to_mul32(input_unit_size);

    uint32_t index_unit_size = index.volume() * index.element_size();
    uint32_t rounded_index_unit_size = round_up_to_mul32(index_unit_size);

    uint32_t output_unit_size = output.get_legacy_shape()[-1] * output.element_size();
    uint32_t rounded_output_unit_size = round_up_to_mul32(output_unit_size);

    auto src_cb_index = CB::c_in0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(rounded_input_unit_size, {{src_cb_index, input_data_format}})
            .set_page_size(src_cb_index, rounded_input_unit_size);
    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src_config);

    auto index_cb_index = CB::c_in1;
    CircularBufferConfig cb_index_config =
        CircularBufferConfig(index_unit_size, {{index_cb_index, index_data_format}})
            .set_page_size(index_cb_index, index_unit_size);
    auto cb_index = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_index_config);

    auto dst_cb_index = CB::c_out0;
    tt::tt_metal::CircularBufferConfig dst_cb_config =
        tt::tt_metal::CircularBufferConfig(rounded_output_unit_size, {{dst_cb_index, output_data_format}})
            .set_page_size(dst_cb_index, rounded_output_unit_size);
    auto batch_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, dst_cb_config);



    bool in_is_dram = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool index_is_dram = index.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    // Create Kernels
    // reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) in_is_dram,
        (std::uint32_t) index_is_dram,
        (std::uint32_t) src_cb_index,
        (std::uint32_t) index_cb_index,
        (std::uint32_t) strides[dim]
    };

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(
            reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) index_cb_index
    };

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // auto cores = grid_to_cores(num_cores_x*num_cores_y, num_cores_x, num_cores_y, false);


    uint32_t unit_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core(i / num_cores_y, i % num_cores_y);

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                input.buffer()->address(),
                index.buffer()->address(),
                u.u32,
                input_unit_size,
                index_unit_size
            }
        );
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tiles_per_core,
                unit_offset
            }
        );

        unit_offset += num_tiles_per_core;
    }

    return {
        std::move(program),
        {reader_kernel_id, writer_kernel_id, num_cores_x, num_cores_y}};
}

void IndexFillOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    // const auto& batch_ids = tensor_args.batch_ids;
    // const auto& input_a = tensor_args.input_a;
    // const auto& input_b = tensor_args.input_b;
    // auto& output_tensor = tensor_return_value;

    // auto batch_ids_buffer = batch_ids.buffer();
    // auto src_buffer_a = input_a.buffer();
    // auto src_buffer_b = input_b.buffer();
    // auto dst_buffer = output_tensor.buffer();

    // {
    //     auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, CoreCoord{0, 0});
    //     runtime_args[0] = batch_ids_buffer->address();
    //     runtime_args[2] = src_buffer_a->address();
    //     runtime_args[3] = src_buffer_b->address();
    // }

    // {
    //     auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
    //     runtime_args[0] = dst_buffer->address();
    // }
}

}  // namespace ttnn::operations::examples
