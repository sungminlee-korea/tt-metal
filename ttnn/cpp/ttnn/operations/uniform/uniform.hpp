
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::uniform {

struct Uniform {
    static Tensor invoke(
        const Tensor& input_tensor,
        const float from,
        const float to,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::uniform

namespace ttnn {
constexpr auto uniform = ttnn::register_operation_with_auto_launch_op<"ttnn::uniform", operations::uniform::Uniform>();
}  // namespace ttnn
