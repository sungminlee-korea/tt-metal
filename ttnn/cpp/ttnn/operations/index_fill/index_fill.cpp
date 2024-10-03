// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"
#include "ttnn/operations/index_fill/device/index_fill_device_operation.hpp"
#include "index_fill.hpp"

namespace ttnn::operations::index_fill {


    Tensor IndexFill::invoke(
        const Tensor &input,
        const Tensor &index,
        const uint32_t dim,
        const std::variant<float, int> value,
        const std::optional<Tensor> &output,
        const std::optional<MemoryConfig> &memory_config) {
            return ttnn::prim::index_fill(input, index, dim, value, output, memory_config);
        }

}  // namespace ttnn::operations::index_fill
