// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear {
struct MorehLinear {
    static std::optional<Tensor> invoke(
        const Tensor& input,
        const Tensor& weight,
        const std::optional<Tensor>& bias,
        std::optional<Tensor> output,
        const std::optional<MemoryConfig>& output_mem_config,
        const DeviceComputeKernelConfig compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_linear

namespace ttnn {
constexpr auto moreh_linear =
    ttnn::register_operation<"ttnn::moreh_linear", ttnn::operations::moreh::moreh_linear::MorehLinear>();
}
