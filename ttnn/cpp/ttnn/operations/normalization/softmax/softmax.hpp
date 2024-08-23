// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

#include "device/softmax_types.hpp"

namespace ttnn {
namespace operations::normalization {

struct SoftmaxOperation {
    // softmax
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int dim_arg,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ScaleMaskSoftmaxOperation {
    // scale_mask_softmax
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const bool is_causal_mask = false,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct SoftmaxInPlaceOperation {

    // softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ScaleMaskSoftmaxInPlaceOperation {

    // scale_mask_softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const bool is_causal_mask = false,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ScaleCausalMaskHWSoftmaxInPlaceOperation {

    // scale_causal_mask_hw_dims_softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto softmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::softmax", ttnn::operations::normalization::SoftmaxOperation>();
constexpr auto scale_mask_softmax = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scale_mask_softmax",
    ttnn::operations::normalization::ScaleMaskSoftmaxOperation>();
constexpr auto softmax_in_place = ttnn::register_operation_with_auto_launch_op<
    "ttnn::softmax_in_place",
    ttnn::operations::normalization::SoftmaxInPlaceOperation>();
constexpr auto scale_mask_softmax_in_place = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scale_mask_softmax_in_place",
    ttnn::operations::normalization::ScaleMaskSoftmaxInPlaceOperation>();
constexpr auto scale_causal_mask_hw_dims_softmax_in_place = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scale_causal_mask_hw_dims_softmax_in_place",
    ttnn::operations::normalization::ScaleCausalMaskHWSoftmaxInPlaceOperation>();

}  // namespace ttnn
