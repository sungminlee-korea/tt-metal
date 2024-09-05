// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "third_party/json/json.hpp"

#include <vector>

namespace ttnn::graph {

std::vector<uint32_t> extract_circular_buffer_allocations_per_core(const nlohmann::json& trace);
std::vector<uint32_t> extract_l1_buffer_allocations(const nlohmann::json& trace);

uint32_t extract_peak_L1_memory_usage(const nlohmann::json& trace);

// Returns count of intermediate and output tensors
std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace);

std::vector<std::string> extract_calltrace(const nlohmann::json& trace);

std::unordered_set<uint32_t> extract_output_tensors(const nlohmann::json& trace);

struct TensorInfo {
    ttnn::Shape shape;
    uint32_t size = 0;
    tt::tt_metal::BufferType type = tt::tt_metal::BufferType::DRAM;

    bool operator==(const TensorInfo& other) const = default;
};

std::ostream &operator<<(std::ostream &os, const TensorInfo &info);

std::vector<TensorInfo> extract_output_info(const nlohmann::json& trace);

} // namespace ttnn::graph
