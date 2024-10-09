// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_config.h"
#include "noc/noc_parameters.h"
#include "llrt/hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

#if defined (ARCH_GRAYSKULL)

#include "hw/inc/grayskull/core_config.h"
#include "hw/inc/grayskull/dev_mem_map.h"
#include "hw/inc/grayskull/eth_l1_address_map.h" // TODO remove when commonruntimeaddressmap is gone
#include "hostdevcommon/common_runtime_address_map.h"
#include "hw/inc/dev_msgs.h"

#include <magic_enum.hpp>
#include <numeric>

#endif

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

void Hal::initialize_gs() {
#if defined (ARCH_GRAYSKULL)

    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));

    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::COUNT));
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)] = ((L1_KERNEL_CONFIG_BASE + L1_KERNEL_CONFIG_SIZE - 1) | (max_alignment - 1)) + 1;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::COUNT));
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)] = MEM_L1_SIZE - mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);

    std::vector<std::vector<uint8_t>> processor_classes(NumTensixDispatchClasses);
    std::vector<uint8_t> processor_types;
    for (auto dispatch_class_idx = 0; dispatch_class_idx < processor_classes.size(); dispatch_class_idx++) {
        HalProcessorClassType dispatch_class = magic_enum::enum_value<HalProcessorClassType>(dispatch_class_idx);
        uint32_t num_processors = dispatch_class == HalProcessorClassType::COMPUTE ? 3 : 1;
        processor_types.resize(num_processors);
        std::iota(processor_types.begin(), processor_types.end(), 0);
        processor_classes[dispatch_class_idx] = processor_types;
    }

    this->core_info_.push_back({HalProgrammableCoreType::TENSIX, CoreType::WORKER, processor_classes, mem_map_bases, mem_map_sizes, true});

    this->mem_alignments_.resize(utils::underlying_type<HalMemType>(HalMemType::COUNT));
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::L1)] = L1_ALIGNMENT;
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::DRAM)] = DRAM_ALIGNMENT;
    this->mem_alignments_[utils::underlying_type<HalMemType>(HalMemType::HOST)] = PCIE_ALIGNMENT;
#endif
}

}  // namespace tt_metal
}  // namespace tt
