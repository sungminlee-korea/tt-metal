// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <csignal>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"  // TTNN_TENSOR_PRINT_PROFILE
#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/operation.hpp"
#include "ttnn/config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

using tt::tt_metal::operation::OptionalConstTensors;
using tt::tt_metal::operation::OptionalTensors;
using tt::tt_metal::operation::Tensors;

using tt::tt_metal::any_tensor_on_multi_device;
using tt::tt_metal::is_tensor_on_device;
using tt::tt_metal::is_tensor_on_device_or_multidevice;
using tt::tt_metal::is_tensor_on_multi_device;
}  // namespace ttnn

namespace ttnn {

namespace core {

inline std::uint32_t pad_to_multiple_of_tile_size(std::uint32_t value) {
    return (value + (ttnn::TILE_SIZE - 1)) / ttnn::TILE_SIZE * ttnn::TILE_SIZE;
}

inline bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type) {
    return tensor.storage_type() == storage_type;
}

inline std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor) {
    if (not tensor.is_allocated() or not is_tensor_on_device_or_multidevice(tensor)) {
        return std::nullopt;
    }
    return tensor.memory_config();
}

inline void set_printoptions(const std::string& profile) {
    tt::tt_metal::tensor_impl::TTNN_TENSOR_PRINT_PROFILE =
        magic_enum::enum_cast<tt::tt_metal::tensor_impl::TensorPrintProfile>(profile, [](char lhs, char rhs) {
            return std::tolower(lhs) == std::tolower(rhs);
        }).value();
}

inline void segfault_handler(int sig) {
    std::cerr << tt::assert::backtrace_to_string() << std::endl;
    exit(EXIT_FAILURE);
}

inline void dump_stack_trace_on_segfault() {
    if (std::signal(SIGSEGV, segfault_handler) == SIG_ERR) {
        std::cerr << "Error: cannot handle SIGSEGV" << std::endl;
        exit(EXIT_FAILURE);
    }
}

}  // namespace core

using core::get_memory_config;
using core::has_storage_type_of;
using core::pad_to_multiple_of_tile_size;
using core::set_printoptions;
}  // namespace ttnn
