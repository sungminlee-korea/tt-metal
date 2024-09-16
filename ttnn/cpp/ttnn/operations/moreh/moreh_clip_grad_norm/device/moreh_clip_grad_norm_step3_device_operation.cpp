// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step3_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

void MorehClipGradNormStep3Operation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& inputs = tensor_args.inputs;
    for (const auto& input : inputs)
        tt::operations::primary::check_tensor(input, "moreh_clip_grad_norm_step3", "input");
    tt::operations::primary::check_tensor(
        tensor_args.clip_coef_clamped, "moreh_clip_grad_norm_step3", "clip_coef_clamped");
}

MorehClipGradNormStep3Operation::program_factory_t MorehClipGradNormStep3Operation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehClipGradNormStep3Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehClipGradNormStep3Operation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehClipGradNormStep3Operation::shape_return_value_t MorehClipGradNormStep3Operation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
};

MorehClipGradNormStep3Operation::tensor_return_value_t MorehClipGradNormStep3Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
}

std::tuple<MorehClipGradNormStep3Operation::operation_attributes_t, MorehClipGradNormStep3Operation::tensor_args_t>
MorehClipGradNormStep3Operation::invoke(const std::vector<Tensor>& inputs, const Tensor& clip_coef_clamped) {
    return {
        operation_attributes_t{},
        tensor_args_t{
            inputs,
            clip_coef_clamped,
        },
    };
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
