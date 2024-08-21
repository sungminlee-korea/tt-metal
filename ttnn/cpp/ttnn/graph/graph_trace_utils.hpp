// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "third_party/json/json.hpp"

#include <vector>

namespace ttnn::graph {

uint32_t extract_peak_memory_usage(const nlohmann::json& trace);

// Returns count of intermediate and output tensors
std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace);

std::vector<std::string> extract_calltrace(const nlohmann::json& trace);

std::unordered_set<uint32_t> extract_output_tensors(const nlohmann::json& trace);

struct TensorInfo {
    ttnn::Shape shape;
    uint32_t size = 0;
    tt::tt_metal::BufferType type = tt::tt_metal::BufferType::DRAM;
};

vector<TensorInfo> extract_output_info(const nlohmann::json& trace);

} // namespace ttnn::graph
