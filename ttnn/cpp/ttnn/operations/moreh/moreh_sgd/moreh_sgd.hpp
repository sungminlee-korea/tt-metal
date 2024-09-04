// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/moreh/moreh_sgd/device/moreh_sgd_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_sgd {
struct MorehSgd {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& param_in,
        const Tensor& grad,
        const std::optional<const Tensor> momentum_buffer_in,
        const std::optional<const Tensor> param_out,
        const std::optional<const Tensor> momentum_buffer_out,
        float lr,
        float momentum,
        float dampening,
        float weight_decay,
        bool nesterov,
        bool momentum_initialized,
        const std::optional<MemoryConfig>& param_out_mem_config,
        const std::optional<MemoryConfig>& momentum_buffer_out_mem_config,
        const DeviceComputeKernelConfig compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_sgd

namespace ttnn {
constexpr auto moreh_sgd =
    ttnn::register_operation<"ttnn::moreh_sgd", ttnn::operations::moreh::moreh_sgd::MorehSgd>();
}
