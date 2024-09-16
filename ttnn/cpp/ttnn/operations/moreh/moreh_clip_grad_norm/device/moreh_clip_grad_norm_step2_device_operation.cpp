// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step2_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

void MorehClipGradNormStep2Operation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    tt::operations::primary::check_tensor(tensor_args.tmp_pow_sum, "moreh_clip_grad_norm_step2", "tmp_pow_sum");
    tt::operations::primary::check_tensor(tensor_args.total_norm, "moreh_clip_grad_norm_step2", "total_norm");
}

MorehClipGradNormStep2Operation::program_factory_t MorehClipGradNormStep2Operation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehClipGradNormStep2Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehClipGradNormStep2Operation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehClipGradNormStep2Operation::shape_return_value_t MorehClipGradNormStep2Operation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
};

MorehClipGradNormStep2Operation::tensor_return_value_t MorehClipGradNormStep2Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
}

std::tuple<MorehClipGradNormStep2Operation::operation_attributes_t, MorehClipGradNormStep2Operation::tensor_args_t>
MorehClipGradNormStep2Operation::invoke(const Tensor& tmp_pow_sum, float norm_type, const Tensor& total_norm) {
    return {
        operation_attributes_t{
            norm_type,
        },
        tensor_args_t{
            tmp_pow_sum,
            total_norm,
        },
    };
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
