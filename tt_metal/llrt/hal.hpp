// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains the TT Hardware Abstraction Layer interface
// This layer abstracts which TT chip is running from the higher
// level APIs
//

#include <cstdint>
#include <vector>
#include <memory>
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/utils.hpp"

enum class CoreType;

namespace tt {

enum class ARCH;

namespace tt_metal {

enum class HalProgrammableCoreType {
    TENSIX     = 0,
    ACTIVE_ETH = 1,
    IDLE_ETH   = 2,
    COUNT      = 3
};

enum class HalProcessorClassType {
    DM0     = 0,
    DM1     = 1,
    COMPUTE = 2,
    COUNT   = 3
};

enum class HalProcessorType {
    // Data movement processors
    DM = 0,

    // Compute processors
    MATH0 = 0, // UNPACK?
    MATH1 = 1, // MATH?
    MATH2 = 2 // PACK?
};

enum class HalL1MemAddrType : uint8_t {
    BARRIER = 0,
    LAUNCH = 1,
    WATCHER = 2,
    DPRINT = 3,
    PROFILER = 4,
    KERNEL_CONFIG = 5,
    UNRESERVED = 6,
    CORE_INFO = 7,
    GO_MSG = 8,
    LAUNCH_MSG_BUFFER_RD_PTR = 9,
    COUNT = 10
};

enum class HalMemType : uint8_t {
    L1 = 0,
    DRAM = 1,
    HOST = 2,
    COUNT = 3
};

using DeviceAddr = std::uint64_t;

class Hal;

/*
    for (auto core_type : programmable_core_types) {
        auto dispatch_classes = hal_3d_thing[core_type]
        for (auto dispatch_class : dispatch_classes) {
            auto processor_types = dispatch_classes[dispatch_class]
            for (auto processor_type : processor_types) {
                // do stuff
            }
        }
    }

*/

// Core information instanced once per core type
class HalCoreInfoType {
    friend class Hal;

  private:
    HalProgrammableCoreType programmable_core_type_;
    CoreType core_type_;
    std::vector<std::vector<uint8_t>> processor_classes_; // processor_class_types_?
    std::vector<DeviceAddr> mem_map_bases_;
    std::vector<uint32_t> mem_map_sizes_;
    bool supports_cbs_;

  public:
    HalCoreInfoType(HalProgrammableCoreType programmable_core_type, CoreType core_type, const std::vector<std::vector<uint8_t>> &processor_classes,
        const std::vector<DeviceAddr>& mem_map_bases, const std::vector<uint32_t>& mem_map_sizes, bool supports_cbs);

    template <typename T = DeviceAddr>
    T get_dev_addr(HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalL1MemAddrType addr_type) const;
    uint32_t get_processor_classes_count() const;
};

template <typename T>
inline T HalCoreInfoType::get_dev_addr(HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalL1MemAddrType>(addr_type);
    TT_ASSERT(index < this->mem_map_bases_.size());
    return reinterpret_cast<T>(this->mem_map_bases_[index]);
}

inline uint32_t HalCoreInfoType::get_dev_size(HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalL1MemAddrType>(addr_type);
    TT_ASSERT(index < this->mem_map_sizes_.size());
    return this->mem_map_sizes_[index];
}

inline uint32_t HalCoreInfoType::get_processor_classes_count() const {
    return this->processor_classes_.size();
}

class Hal {
  private:
    std::mutex lock;
    bool initialized_;
    std::vector<HalCoreInfoType> core_info_;
    std::vector<uint32_t> mem_alignments_;

    void initialize_gs();
    void initialize_wh();
    void initialize_bh();

  public:
    Hal();

    void initialize(tt::ARCH arch);

    uint32_t get_programmable_core_type_count() const;
    HalProgrammableCoreType get_programmable_core_type(uint32_t core_type_index) const;
    uint32_t get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const;
    CoreType get_core_type(uint32_t programmable_core_type_index) const;
    uint32_t get_processor_classes_count(std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type) const;

    template <typename T = DeviceAddr>
    T get_dev_addr(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const;
    template <typename T = DeviceAddr>
    T get_dev_addr(uint32_t programmable_core_type_index, HalL1MemAddrType addr_type) const;
    uint32_t get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const;

    uint32_t get_alignment(HalMemType memory_type) const;

    bool get_supports_cbs(uint32_t programmable_core_type_index) const;
};

inline uint32_t Hal::get_programmable_core_type_count() const {
    return core_info_.size();
}

inline uint32_t Hal::get_processor_classes_count(std::variant<HalProgrammableCoreType, uint32_t> programmable_core_type) const {
    return std::visit(
        [&](auto &&core_type_specifier) -> uint32_t {
            using T = std::decay_t<decltype(core_type_specifier)>;
            uint32_t index = this->core_info_.size();
            if constexpr (std::is_same_v<T, HalProgrammableCoreType>) {
                index = utils::underlying_type<HalProgrammableCoreType>(core_type_specifier);
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                index = core_type_specifier;
            }
            TT_ASSERT(index < this->core_info_.size());
            return this->core_info_[index].get_processor_classes_count();
        },
    programmable_core_type);
}

inline HalProgrammableCoreType Hal::get_programmable_core_type(uint32_t core_type_index) const {
    return core_info_[core_type_index].programmable_core_type_;
}

inline CoreType Hal::get_core_type(uint32_t core_type_index) const {
    return core_info_[core_type_index].core_type_;
}

template <typename T>
inline T Hal::get_dev_addr(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_addr<T>(addr_type);
}

template <typename T>
inline T Hal::get_dev_addr(uint32_t programmable_core_type_index, HalL1MemAddrType addr_type) const {
    TT_ASSERT(programmable_core_type_index < this->core_info_.size());
    return this->core_info_[programmable_core_type_index].get_dev_addr<T>(addr_type);
}

inline uint32_t Hal::get_dev_size(HalProgrammableCoreType programmable_core_type, HalL1MemAddrType addr_type) const {
    uint32_t index = utils::underlying_type<HalProgrammableCoreType>(programmable_core_type);
    TT_ASSERT(index < this->core_info_.size());
    return this->core_info_[index].get_dev_size(addr_type);
}

inline uint32_t Hal::get_alignment(HalMemType memory_type) const {
    uint32_t index = utils::underlying_type<HalMemType>(memory_type);
    TT_ASSERT(index < this->mem_alignments_.size());
    return this->mem_alignments_[index];
}

inline bool Hal::get_supports_cbs(uint32_t programmable_core_type_index) const {
    return this->core_info_[programmable_core_type_index].supports_cbs_;
}

extern Hal hal;

}  // namespace tt_metal
}  // namespace tt
