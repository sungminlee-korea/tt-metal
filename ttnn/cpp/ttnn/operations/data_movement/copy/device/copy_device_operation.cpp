// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement {

void CopyDeviceOperation::validate_with_output_tensors(const std::vector<Tensor>& input_tensors,
                                                       const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::BFLOAT8_B or
                 input_tensor_a.get_dtype() == DataType::FLOAT32,
             "Typecast operation is only supported on Grayskull for float/bfloat inputs");
    TT_FATAL(this->output_dtype == DataType::BFLOAT16 or this->output_dtype == DataType::BFLOAT8_B or
                 this->output_dtype == DataType::FLOAT32,
             "Typecast operation is only supported on Grayskull for float/bfloat outputs");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
             "Copy does not currently support sharding");
    if (input_tensors.size() == 2) {
        const auto& dst_tensor = input_tensors[1];
        TT_FATAL(input_tensor_a.get_legacy_shape() == dst_tensor.get_legacy_shape(), "Error");
        TT_FATAL(input_tensor_a.get_layout() == dst_tensor.get_layout(), "Error");
        TT_FATAL(input_tensor_a.memory_config().memory_layout == dst_tensor.memory_config().memory_layout, "Error");
        TT_FATAL(dst_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
                 "Copy does not currently support sharding");
    }
    DataType output_dtype = this->output_dtype;
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        const auto& out_tensor = output_tensors.at(0).value();
        const auto& output_shape_tensor = input_tensors.size() == 2 ? input_tensors[1] : input_tensor_a;
        TT_FATAL(out_tensor.get_logical_shape() == output_shape_tensor.get_logical_shape() &&
                     out_tensor.get_padded_shape() == output_shape_tensor.get_padded_shape(),
                 "The input tensors need a shape of {}/{}, however the output tensor is only {}/{}",
                 output_shape_tensor.get_logical_shape(),
                 output_shape_tensor.get_padded_shape(),
                 out_tensor.get_logical_shape(),
                 out_tensor.get_padded_shape());
        output_dtype = out_tensor.get_dtype();
    }
    if (output_dtype != input_tensor_a.get_dtype()) {
        TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Only tile layout supports dtype conversion");
    }
    auto out_mem_config = (!output_tensors.empty() && output_tensors.at(0).has_value())
                              ? output_tensors.at(0).value().memory_config()
                              : this->output_mem_config;
    TT_FATAL(out_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
             "Copy does not currently support sharding");
}

std::vector<ttnn::SimpleShape> CopyDeviceOperation::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    if (input_tensors.size() == 2) {
        return {input_tensors[1].get_logical_shape()};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        return {input_tensor.get_logical_shape()};
    }
}

std::vector<Tensor> CopyDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    if (input_tensors.size() == 2) {
        return {input_tensors[1]};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        std::vector<Tensor> output_tensors;
        output_tensors.emplace_back(create_device_tensor(input_tensor.get_legacy_shape(),
                                                         output_dtype,
                                                         input_tensors.at(0).get_layout(),
                                                         input_tensor.device(),
                                                         output_mem_config));
        return output_tensors;
    }
}

operation::ProgramWithCallbacks CopyDeviceOperation::create_program(const std::vector<Tensor>& input_tensors,
                                                                    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);

    switch (CopyDeviceOperation::get_parallelization_strategy(input_tensors)) {
        case CopyOpParallelizationStrategy::MULTI_CORE:
        default: return copy_multi_core(input_tensor, output_tensor);
    }
}

CopyOpParallelizationStrategy CopyDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return CopyOpParallelizationStrategy::MULTI_CORE;
}

}  // namespace ttnn::operations::data_movement
