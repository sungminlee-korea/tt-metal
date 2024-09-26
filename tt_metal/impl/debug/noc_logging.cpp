// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_logging.hpp"

// 32 buckets to match the number of bits in uint32_t lengths on device
#define NOC_DATA_SIZE 32
typedef noc_data_t std::array<uint64_t, NOC_DATA_SIZE>;

void DumpCoreNocData(CoreDescriptor *logical_core, noc_data_t &noc_data) {
}

void DumpDeviceNocData(Device *device) {
    // Need to treat dispatch cores and normal cores separately, so keep track of which cores are dispatch.
    map<CoreType, set<CoreCoord>> physical_printable_dispatch_cores;
    unsigned num_cqs = tt::llrt::OptionsG.get_num_hw_cqs();
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    for (auto logical_core : tt::get_logical_dispatch_cores(device->id(), num_cqs, dispatch_core_type)) {
        CoreCoord physical_core = device->physical_core_from_logical_core(logical_core, dispatch_core_type);
        physical_printable_dispatch_cores[dispatch_core_type].insert(physical_core);
    }

    // Now go through all cores on the device, and dump noc data for them.
    CoreCoord logical_grid_size = device->logical_grid_size();
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_coord(x, y);
            CoreCoord worker_core = device->worker_core_from_logical_core(logical_coord);
            all_physical_printable_cores[CoreType::WORKER].insert(worker_core);
        }
    }
    for (const auto &eth_core : device->get_active_ethernet_cores()) {
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        all_physical_printable_cores[CoreType::ETH].insert(physical_core);
    }
    for (const auto &eth_core : device->get_inactive_ethernet_cores()) {
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        all_physical_printable_cores[CoreType::ETH].insert(physical_core);
    }
}

void DumpNocData(std::vector<Device *> devices) {
    for (Device *device : devices) {
    }
}
