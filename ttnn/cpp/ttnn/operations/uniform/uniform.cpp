
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform.hpp"

#include "device/uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

Tensor Uniform::invoke(
    const Tensor &input_tensor,
    const float from,
    const float to,
    const std::optional<MemoryConfig> &memory_config,
    const std::optional<DeviceComputeKernelConfig> &compute_kernel_config) {
    return ttnn::prim::uniform(input_tensor, from, to, memory_config, compute_kernel_config);
}

}  // namespace ttnn::operations::uniform
