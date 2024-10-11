// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common_values.hpp"
#include "dev_mem_map.h"
#include "noc/noc_parameters.h"
#include "hostdevcommon/profiler_common.h"

/*
* This file contains addresses that are visible to both host and device compiled code.
*/

// TODO: move these out of the memory map into profiler code
constexpr static std::uint32_t PROFILER_OP_SUPPORT_COUNT = 1000;
constexpr static std::uint32_t PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC = kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * (kernel_profiler::PROFILER_L1_PROGRAM_ID_COUNT +  kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT + kernel_profiler::PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT) * PROFILER_OP_SUPPORT_COUNT;
constexpr static std::uint32_t PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC * sizeof(uint32_t);

static_assert (PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC > kernel_profiler::PROFILER_L1_BUFFER_SIZE);

// Kernel config buffer is WIP
// Size is presently based on the old sizes of the RTAs + CB config + Sems
// plus some extra space freed up in the mem map
constexpr static std::uint32_t L1_KERNEL_CONFIG_BASE = MEM_MAP_END;
constexpr static std::uint32_t L1_KERNEL_CONFIG_SIZE = 4 * 1024 + 256 + 128 + 512;

constexpr static std::uint32_t IDLE_ERISC_L1_KERNEL_CONFIG_BASE = MEM_IERISC_MAP_END;

constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;

// Helper functions to convert NoC coordinates to NoC-0 coordinates, used in metal as "physical" coordinates.
#define NOC_0_X(noc_index, noc_size_x, x) (noc_index == 0 ? (x) : (noc_size_x-1-(x)))
#define NOC_0_Y(noc_index, noc_size_y, y) (noc_index == 0 ? (y) : (noc_size_y-1-(y)))

static_assert(L1_KERNEL_CONFIG_BASE % L1_ALIGNMENT == 0);
