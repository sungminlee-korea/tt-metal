// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

using namespace tt::tt_metal;
class CommandQueueFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    Device* device_;
    Device *mmio_device_;
    uint32_t pcie_id;

    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        this->mmio_device_ = tt::tt_metal::CreateDevice(0);

        const int device_id =7;
        this->device_ = tt::tt_metal::CreateDevice(device_id);
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    }

    void TearDown() override {
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        tt::tt_metal::CloseDevice(this->mmio_device_);
        tt::tt_metal::CloseDevice(this->device_);
    }
};


class CommandQueueMultiDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

       std::vector<int> chip_ids = {0,7};
       for (const auto &id: chip_ids) {
///        for (unsigned int id = 0; id < num_devices_; id++) {
            auto* device = tt::tt_metal::CreateDevice(id);
            devices_.push_back(device);
        }
        // while (true) {}
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    }

    void TearDown() override {
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};

class CommandQueuePCIDevicesFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumPCIeDevices();
        if (num_devices_ < 2) {
            GTEST_SKIP();
        }

        for (unsigned int id = 0; id < num_devices_; id++) {
            auto* device = tt::tt_metal::CreateDevice(id);
            devices_.push_back(device);
        }
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    }

    void TearDown() override {
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};
