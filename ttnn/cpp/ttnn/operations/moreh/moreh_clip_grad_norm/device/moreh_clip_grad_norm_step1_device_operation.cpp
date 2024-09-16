// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step1_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

void MorehClipGradNormStep1Operation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& inputs = tensor_args.inputs;
    for (const auto& input : inputs)
        tt::operations::primary::check_tensor(input, "moreh_clip_grad_norm_step1", "input");
    tt::operations::primary::check_tensor(tensor_args.tmp_pow_sum, "moreh_clip_grad_norm_step1", "tmp_pow_sum");
}

MorehClipGradNormStep1Operation::program_factory_t MorehClipGradNormStep1Operation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehClipGradNormStep1Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehClipGradNormStep1Operation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehClipGradNormStep1Operation::shape_return_value_t MorehClipGradNormStep1Operation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
};

MorehClipGradNormStep1Operation::tensor_return_value_t MorehClipGradNormStep1Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
}

std::tuple<MorehClipGradNormStep1Operation::operation_attributes_t, MorehClipGradNormStep1Operation::tensor_args_t>
MorehClipGradNormStep1Operation::invoke(
    const std::vector<Tensor>& inputs,
    float norm_type,
    uint32_t tile_offset_of_tmp_pow_sum,
    const Tensor& tmp_pow_sum) {
    return {
        operation_attributes_t{
            norm_type,
            tile_offset_of_tmp_pow_sum,
        },
        tensor_args_t{
            inputs,
            tmp_pow_sum,
        },
    };
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
