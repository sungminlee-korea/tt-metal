// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/avgpool/avg_pool.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace tt {
namespace tt_metal {

template<PoolType pool>
Tensor pool_2d(const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto input_shape = input.get_legacy_shape();
    switch (pool) {
        case PoolType::AVG: {
            uint32_t height_without_padding = input.get_logical_shape()[-2];
            return ttnn::sum(input, int(input_shape.rank() - 2), true, memory_config, std::nullopt, 1 / float(height_without_padding));
        }
        default:
            TT_ASSERT(false && "Undefined pool type");
    }
}

Tensor avg_pool2d(const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto output = input;

    tt::tt_metal::LegacyShape in_shape = input.get_legacy_shape();
    auto output_shape = ttnn::SimpleShape({in_shape[0], 1, in_shape[1] * in_shape[2], in_shape[3]});
    output = output.reshape(output_shape);

    output = pool_2d<PoolType::AVG>(output, memory_config, output_dtype);
    return output;
}

}  // namespace tt_metal
}  // namespace tt
