// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/moreh_linear_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {
struct MorehLinearBackward {
    static std::tuple<bool, bool, bool> get_required_outputs(const std::vector<bool>& are_required_outputs);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& weight,
        std::vector<bool> &are_required_outputs,
        const std::optional<const Tensor> bias,
        const std::optional<const Tensor> input_grad,
        const std::optional<const Tensor> weight_grad,
        const std::optional<const Tensor> bias_grad,
        const std::optional<ttnn::MemoryConfig>& input_grad_mem_config,
        const std::optional<ttnn::MemoryConfig>& weight_grad_mem_config,
        const std::optional<ttnn::MemoryConfig>& bias_grad_mem_config,
        const DeviceComputeKernelConfig compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_linear_backward

namespace ttnn {
constexpr auto moreh_linear_backward =
    ttnn::register_operation<"ttnn::moreh_linear_backward", ttnn::operations::moreh::moreh_linear_backward::MorehLinearBackward>();
}
