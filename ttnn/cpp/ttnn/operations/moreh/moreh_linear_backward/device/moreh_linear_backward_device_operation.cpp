// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_backward_device_operation.hpp"

#include <cstdint>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

void MorehBiasAddBackwardOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& output_grad = tensor_args.output_grad;
    auto& input = tensor_args.input;
    auto& weight = tensor_args.weight;
    auto& input_grad = tensor_args.input_grad;
    auto& weight_grad = tensor_args.weight_grad;
    auto& bias_grad = tensor_args.bias_grad;

    if (input_grad.has_value()) {
        auto input_shape = input.get_shape();
        auto input_grad_shape = input_grad->get_shape();
        TT_ASSERT(input_shape == input_grad_shape, "both tensors should be the same shape");
    }

    if (weight_grad.has_value()) {
        auto weight_shape = weight.get_shape();
        auto weight_grad_shape = weight_grad->get_shape();
        TT_ASSERT(weight_shape == weight_grad_shape, "both tensors should be the same shape");
    }

    if (bias_grad.has_value()) {
        auto bias_grad_shape = bias_grad->get_shape();
        auto bias_grad_tensor = bias_grad.value();
        TT_ASSERT(
            tt::operations::primary::is_scalar(bias_grad_tensor) ||
                tt::operations::primary::is_1d_tensor(bias_grad_tensor),
            "bias_grad tensor should be 1d or scalar");
    }
}

// moreh_linear_backward_validate
// void MorehBiasAddBackwardOperation::validate_inputs(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
//     auto& output_grad = tensor_args.output_grad;
//     auto& input = tensor_args.input;
//     auto& weight = tensor_args.weight;
//     auto& input_grad = tensor_args.input_grad;
//     auto& weight_grad = tensor_args.weight_grad;
//     auto& bias_grad = tensor_args.bias_grad;

//     if (input_grad.has_value()) {
//         auto input_shape = input.get_shape();
//         auto input_grad_shape = input_grad->get_shape();
//         TT_ASSERT(input_shape == input_grad_shape, "both tensors should be the same shape")
//     }

//     if (weight_grad.has_value()) {
//         auto weight_shape = weight.get_shape();
//         auto weight_grad_shape = weight_grad->get_shape();
//         TT_ASSERT(weight_shape == weight_grad_shape, "both tensors should be the same shape")
//     }

//     if (bias_grad.has_value()) {
//         auto bias_grad_shape = bias_grad->get_shape();
//         auto bias_grad_tensor = bias_grad.value();
//         TT_ASSERT(tt::operations::primary::is_scalar(bias_grad_tensor) ||
//         tt::operations::primary::is_1d_tensor(bias_grad_tensor), "bias_grad tensor should be 1d or scalar")
//     }
// }

MorehBiasAddBackwardOperation::program_factory_t MorehBiasAddBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& bias_grad = tensor_args.bias_grad.value();
    if (tt::operations::primary::is_scalar(bias_grad))
        return SingleCoreProgramFactory();
    return MultiCoreProgramFactory();
    // return tt::operations::primary::is_scalar(bias_grad) ?
    // static_cast<program_factory_t>(SingleCoreProgramFactory{}) :
    // static_cast<program_factory_t>(MultiCoreProgramFactory{});
}

void MorehBiasAddBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehBiasAddBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehBiasAddBackwardOperation::shape_return_value_t MorehBiasAddBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor_shape = tensor_args.bias.value().get_shape();

    return {input_tensor_shape};
};

MorehBiasAddBackwardOperation::tensor_return_value_t MorehBiasAddBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.input.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.input.device();

    std::vector<std::optional<Tensor>> ret;
    auto bias_grad_mem_config = operation_attributes.bias_grad_mem_config;

    if (tensor_args.bias.has_value()) {
        ret.push_back(tensor_args.bias.value());
    } else {
        ret.push_back(create_device_tensor(output_shapes.at(0).value(), dtype, layout, device, bias_grad_mem_config));
    }

    return std::move(ret);
}

std::tuple<MorehBiasAddBackwardOperation::operation_attributes_t, MorehBiasAddBackwardOperation::tensor_args_t>
MorehBiasAddBackwardOperation::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& weight_grad,
    const std::optional<Tensor>& bias_grad,

    const std::optional<ttnn::MemoryConfig>& input_grad_mem_config,
    const std::optional<ttnn::MemoryConfig>& weight_grad_mem_config,
    const std::optional<ttnn::MemoryConfig>& bias_grad_mem_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& ck) {
    return {
        MorehBiasAddBackwardOperation::operation_attributes_t{
            are_required_outputs,
            input_grad_mem_config.value_or(input.memory_config()),
            weight_grad_mem_config.value_or(input.memory_config()),
            bias_grad_mem_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input_grad->device()->arch(), ck)},
        MorehBiasAddBackwardOperation::tensor_args_t{
            output_grad, input, weight, bias, input_grad, weight_grad, bias_grad}};
}
}  // namespace ttnn::operations::moreh::moreh_linear_backward
