// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_llama_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void RotaryEmbeddingLlama::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    const auto& trans_mat = input_tensors.at(3);
    TT_FATAL(input_tensors.size() == 4);
    auto ref_device = input_tensor.device();
    for (const auto& input : input_tensors) {
        TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
        TT_FATAL(input.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
        TT_FATAL(input.device() == ref_device, "Operands to rotary embedding need to be on same device!");
        TT_FATAL((input.get_layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    }

    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
    uint32_t B = input_tensor.get_legacy_shape()[0];
    uint32_t head_dim = input_tensor.get_legacy_shape()[-1];
    TT_FATAL(cos.get_dtype() == sin.get_dtype(), "Cos and Sin dtypes must match");
    TT_FATAL(cos.get_legacy_shape() == sin.get_legacy_shape(), "Cos and Sin dims must match");
    TT_FATAL(cos.get_legacy_shape()[0] == 1 && cos.get_legacy_shape()[1] == 1 && cos.get_legacy_shape()[-1] == head_dim, "Cos dims must match input dims");

    TT_FATAL(trans_mat.get_legacy_shape()[0] == 1 && trans_mat.get_legacy_shape()[1] == 1, "Transformation matrix must have 1st & 2nd dim equal to 1");
    TT_FATAL(trans_mat.get_legacy_shape()[-2] == TILE_HEIGHT, "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
    TT_FATAL(trans_mat.get_legacy_shape()[-1] == TILE_WIDTH, "Transformation matrix must have 4rd dim equal to TILE_WIDTH");


    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
}

std::vector<Shape> RotaryEmbeddingLlama::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.get_legacy_shape();
    return {shape};
}

std::vector<Tensor> RotaryEmbeddingLlama::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = this->compute_output_shapes(input_tensors)[0];
    return {create_device_tensor(
        output_shape, input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), this->output_mem_config)};
}

operation::ProgramWithCallbacks RotaryEmbeddingLlama::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    const auto& trans_mat = input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case RotaryEmbeddingLlamaOpParallelizationStrategy::MULTI_CORE:
            return rotary_embedding_llama_multi_core(input_tensor, cos, sin, trans_mat, output_tensor, this->compute_kernel_config);
            break;
        case RotaryEmbeddingLlamaOpParallelizationStrategy::SINGLE_CORE:
        default: return rotary_embedding_llama_single_core(input_tensor, cos, sin, trans_mat, output_tensor, this->compute_kernel_config);
    }
}

RotaryEmbeddingLlamaOpParallelizationStrategy RotaryEmbeddingLlama::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // num_rows = 1 x 8 x 128 x 128 / 128 / 32 = 32
    uint32_t num_rows = input_tensor.volume() / input_tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
    if (num_rows > 1) {
        return RotaryEmbeddingLlamaOpParallelizationStrategy::MULTI_CORE;
    } else {return RotaryEmbeddingLlamaOpParallelizationStrategy::SINGLE_CORE;}
    return RotaryEmbeddingLlamaOpParallelizationStrategy::SINGLE_CORE;
}

tt::stl::reflection::Attributes RotaryEmbeddingLlama::attributes() const {
    return {
        {"seq_len", this->seq_len},
        {"output_mem_config", this->output_mem_config},
    };
}

const operation::Hash RotaryEmbeddingLlama::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return operation::hash_operation<RotaryEmbeddingLlama>(this->seq_len, this->output_mem_config, input_tensors);
}

}  // namespace tt_metal

}  // namespace tt
