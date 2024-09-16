// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {
struct MorehClipGradNorm {
    static Tensor invoke(
        const std::vector<Tensor>& inputs,
        float max_norm,
        float norm_type,
        bool error_if_nonfinite,
        const std::optional<Tensor>& total_norm,
        const MemoryConfig& memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm

namespace ttnn {
constexpr auto moreh_clip_grad_norm = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_clip_grad_norm",
    ttnn::operations::moreh::moreh_clip_grad_norm::MorehClipGradNorm>();
}
