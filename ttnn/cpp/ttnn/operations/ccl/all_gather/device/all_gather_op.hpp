// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"


#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

using namespace tt::tt_metal;

namespace ttnn {

enum AllGatherBidirectionalMode {
    // Splits the tensor into two and sends each half in opposite directions
    // the full width around the ring
    SPLIT_TENSOR,
    // Doesn't split the tensor and sends the full tensor in both directions,
    // half-way around the ring
    FULL_TENSOR
};

namespace all_gather_op {
using ccl::Topology;
}; // namespace all_gather_op

using ccl::EriscDatamoverBuilder;

class AllGatherConfig {
    static AllGatherBidirectionalMode choose_bidirectional_mode(Tensor const& input_tensor, bool fuse_op);

   public:
    AllGatherConfig(Tensor const& input_tensor, Tensor const& output_tensor, uint32_t dim, uint32_t ring_size, uint32_t num_links, all_gather_op::Topology topology, std::size_t num_buffers_per_worker, bool fuse_op=false, const std::optional<size_t> user_defined_num_workers=std::nullopt);

    uint32_t get_erisc_handshake_address() const { return this->erisc_handshake_address; }

    uint32_t get_num_eth_buffers_per_edm() const { return this->num_eth_buffers; }
    uint32_t get_num_workers_per_link() const { return this->num_workers_per_link; }
    uint32_t get_num_workers() const { return this->num_workers_per_link * this->num_links; }

    uint32_t get_eth_buffer_size() const { return this->eth_buffer_size; }

    uint32_t get_eth_sems_l1_base_byte_address() const { return this->eth_sems_l1_base_byte_address; }

    uint32_t get_eth_buffers_l1_base_byte_address() const { return this->eth_buffers_l1_base_byte_address; }

    uint32_t get_semaphore_size() const { return this->semaphore_size; }
    std::size_t get_num_buffers_per_channel() const { return this->num_edm_buffers_per_channel; }

    uint32_t get_num_edm_channels_in_clockwise_direction() const {
        return this->enable_bidirectional ?
            this->num_workers_per_link / 2 :
            this->num_workers_per_link;
    }
    uint32_t get_ring_size() const { return this->ring_size; }
    bool is_payload_and_channel_sync_merged() const { return enable_merged_payload_and_channel_sync;}
    bool is_buffer_in_clockwise_ring(const uint32_t buffer_index) const {
        // For now we split it as lower half => clockwise, upper half => counter-clockwise
        // This is slightly suboptimal since the non-full-chunks go to the upper half.
        // A more optimal split would be round robin
        return this->enable_bidirectional ?
            buffer_index < get_num_edm_channels_in_clockwise_direction() :
            true;
    }
    AllGatherBidirectionalMode get_bidirectional_mode() const { return this->bidirectional_mode; }
    uint32_t get_num_edm_channels_in_counter_clockwise_direction() const {
        // return all_gather_buffer_params::enable_bidirectional ? all_gather_buffer_params::num_buffers - all_gather_buffer_params::num_buffers / 2 : 0;
        // Force all through counter-clockwise direction
        return this->num_workers_per_link - this->get_num_edm_channels_in_clockwise_direction();
    }

    bool is_input_dram() const { return input_is_dram; }
    bool is_output_dram() const { return output_is_dram; }


    void print() const {
        log_trace(tt::LogOp, "AllGatherConfig: (");
        log_trace(tt::LogOp, "\tis_sharded: {}", is_sharded);
        log_trace(tt::LogOp, "\terisc_handshake_address: {}", erisc_handshake_address);
        log_trace(tt::LogOp, "\tnum_buffers: {}", num_eth_buffers);
        log_trace(tt::LogOp, "\tnum_workers_per_link: {}", num_workers_per_link);
        log_trace(tt::LogOp, "\tnum_edm_buffers_per_channel: {}", num_edm_buffers_per_channel);
        log_trace(tt::LogOp, "\teth_buffer_size: {}", eth_buffer_size);
        log_trace(tt::LogOp, "\tsemaphore_size: {}", semaphore_size);
        log_trace(tt::LogOp, "\tsemaphore_offset: {}", semaphore_offset);
        log_trace(tt::LogOp, "\teth_buffers_l1_base_byte_address: {}", eth_buffers_l1_base_byte_address);
        log_trace(tt::LogOp, "\teth_sems_l1_base_byte_address: {}", eth_sems_l1_base_byte_address);
        log_trace(tt::LogOp, "\tenable_bidirectional: {}", enable_bidirectional);
        log_trace(tt::LogOp, ")");
    }

   private:
    const uint32_t erisc_handshake_address;
    uint32_t ring_size;
    uint32_t num_links;
    uint32_t num_eth_buffers;
    uint32_t num_workers_per_link;
    uint32_t num_edm_buffers_per_channel;
    uint32_t eth_buffer_size;
    uint32_t semaphore_size;
    uint32_t semaphore_offset;
    uint32_t eth_buffers_l1_base_byte_address;
    uint32_t eth_sems_l1_base_byte_address;
    const all_gather_op::Topology topology;
    AllGatherBidirectionalMode bidirectional_mode;
    bool is_sharded;
    bool enable_bidirectional;
    const bool input_is_dram;
    const bool output_is_dram;
    const bool enable_merged_payload_and_channel_sync;
};

struct AllGather {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<size_t> user_defined_num_workers;
    const std::optional<size_t> user_defined_num_buffers_per_channel;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const all_gather_op::Topology topology;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

AllGather create_all_gather_struct(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> user_defined_num_workersm,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<Device*>& devices
);

// All Gather Variants
operation::ProgramWithCallbacks all_gather_full_shard_grid(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology);
operation::ProgramWithCallbacks all_gather_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel);
operation::ProgramWithCallbacks all_gather_multi_core_with_workers_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset = CoreCoord(0, 0));



namespace operations {
namespace ccl {

Tensor all_gather(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
