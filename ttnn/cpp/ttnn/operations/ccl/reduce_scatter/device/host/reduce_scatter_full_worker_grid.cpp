// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_worker_builder.hpp"

// Includes that need to be moved to CCL datastructures header
#include <vector>
#include <algorithm>
#include <limits>
#include <ranges>

using namespace tt::constants;

// Notes on abbreviations:
// cw = clockwise
// ccw = counter-clockwise
// edm = erisc data mover

// How this reduce_scatter op works:
// For each chip, we have a element range of the input tensor shape that will eventually scatter
// out to it. For all other chunks outside that range, the chip will forward the chunk to the next chip.
// While forwarding the data, the chip will also reduce it with the local input tensor chunk corresponding
// with that received chunk. It will forward the partially reduced chunk.
// Reduces along rank

namespace ttnn {

namespace ccl {
namespace reduce_scatter_detail {




static std::size_t decide_number_of_edm_channels(
   ttnn::ccl::CCLOpConfig const& ccl_op_config, std::size_t max_num_workers, bool enable_bidirectional) {
    bool is_linear_topology = ccl_op_config.get_topology() == ttnn::ccl::Topology::Linear;
    TT_ASSERT(!is_linear_topology || max_num_workers > 1);
    if (is_linear_topology) {
        // Workers must be evenly divided for line reduce scatter
        max_num_workers = tt::round_down(max_num_workers, 2);
    }
    return std::min<std::size_t>(max_num_workers, enable_bidirectional || is_linear_topology ? 8 : 4);
}


struct EdmInterfaceAddresses {
    std::unordered_map<int, uint32_t> worker_sender_edm_semaphore_addresses;
    std::unordered_map<int, uint32_t> worker_sender_edm_buffer_addresses;
    std::unordered_map<int, uint32_t> worker_receiver_edm_semaphore_addresses;
    std::unordered_map<int, uint32_t> worker_receiver_edm_buffer_addresses;
};


static std::size_t get_global_worker_id(std::size_t link, std::size_t channel_id, std::size_t num_channels_per_link) {
    return link * num_channels_per_link + channel_id;
}
static std::size_t get_global_worker_id(WorkerAttributes const& attrs, std::size_t num_channels_per_link) {
    return get_global_worker_id(attrs.link, attrs.channel, num_channels_per_link);
}


// Future work: split this up further:
// 1) assign workers to EDM channel (with buffer sharing mode specified too)
// 2) Compute the semaphore and buffer addresses (for each EDM channel and worker)
// For now - the mapping between workers and EDM channels is 1:1
static void add_worker_config_to_edm_builders(
    Device* device,
    RingReduceScatterWrappedTensorSlicer& tensor_slicer,  // TODO: Update to Generic ReduceScatterSlicer when it is implemented
    ccl::CCLOpConfig const& op_config,
    std::vector<WorkerAttributes> const& all_worker_attributes,
    std::size_t num_channels_per_edm,
    std::size_t num_buffers_per_channel,

    std::vector<ttnn::ccl::EriscDatamoverBuilder>& clockwise_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& counter_clockwise_edm_builders,

    std::size_t worker_sender_semaphore_id,
    std::size_t worker_receiver_semaphore_id,
    std::size_t link,
    std::size_t ring_size,
    std::size_t ring_index,

    EdmInterfaceAddresses& edm_interface_addresses) {
    bool is_linear = op_config.get_topology() == ttnn::ccl::Topology::Linear;
    for (std::size_t c = 0; c < num_channels_per_edm; ++c) {
        std::size_t num_workers_per_eth_buffer = 1;
        auto global_worker_index = get_global_worker_id(link, c, num_channels_per_edm);
        TT_ASSERT(global_worker_index < all_worker_attributes.size());
        WorkerAttributes const& worker_attrs = all_worker_attributes[global_worker_index];

        std::vector<ttnn::ccl::WorkerXY> sender_worker_coords;
        std::vector<ttnn::ccl::WorkerXY> receiver_worker_coords;
        auto const& worker_noc_coords = device->worker_core_from_logical_core(worker_attrs.location_logical);
        sender_worker_coords.push_back(ttnn::ccl::WorkerXY(worker_noc_coords.x, worker_noc_coords.y));
        receiver_worker_coords.push_back(ttnn::ccl::WorkerXY(worker_noc_coords.x, worker_noc_coords.y));

        // Get the maximum message size we'd like to use. Not the actual packet size
        // If linear, then we want to reuse the slicer in both directions
        std::size_t global_worker_idx = c + num_channels_per_edm * link;
        log_trace(tt::LogOp, "get_worker_slice_size_bytes");
        std::size_t worker_tensor_slice_index = !is_linear ? global_worker_idx : (c / 2) + (num_channels_per_edm / 2) * link;
        std::size_t expected_message_size_bytes = (num_buffers_per_channel == 1) ? tensor_slicer.get_worker_slice_size_bytes(global_worker_idx)
                                                                           : clockwise_edm_builders.at(link).get_eth_buffer_size_bytes();

        bool is_in_clockwise_direction = worker_attrs.direction == Direction::CLOCKWISE;
        bool is_first_device_in_line = is_linear && ((is_in_clockwise_direction && ring_index == 0) ||
                                                     (!is_in_clockwise_direction && ring_index == ring_size - 1));
        bool is_last_device_in_line = is_linear && ((!is_in_clockwise_direction && ring_index == 0) ||
                                                    (is_in_clockwise_direction && ring_index == ring_size - 1));

        bool sender_enabled = (!is_linear || !is_last_device_in_line); // update for linear
        if (sender_enabled) {
            log_trace(tt::LogOp, "Adding sender EDM channel to {} edm builder", worker_attrs.direction == Direction::CLOCKWISE ? "clockwise" : "counter-clockwise");
            auto& sender_edm_builder = is_in_clockwise_direction ? clockwise_edm_builders.at(link)
                                                                              : counter_clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding sender EDM channel");
            ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                sender_edm_builder.add_sender_channel(
                    worker_sender_semaphore_id,
                    1,  // cw_edm_channel_num_messages_to_send_per_transfer.at(c) * (ring_size - 1),
                    sender_worker_coords,
                    expected_message_size_bytes);
            edm_interface_addresses.worker_sender_edm_semaphore_addresses.insert(
                {global_worker_idx, sender_channel_buffer_info.eth_semaphore_l1_address});
            edm_interface_addresses.worker_sender_edm_buffer_addresses.insert(
                {global_worker_idx, sender_channel_buffer_info.eth_buffer_l1_address});
            log_trace(tt::LogOp, "\tAdded");
        }

        bool receiver_enabled = (!is_linear || !is_first_device_in_line);
        if (receiver_enabled) {
            log_trace(tt::LogOp, "Adding receiver EDM channel to {} edm builder", worker_attrs.direction == Direction::CLOCKWISE ? "counter-clockwise" : "clockwise");
            auto& receiver_edm_builder =
                is_in_clockwise_direction ? counter_clockwise_edm_builders.at(link) : clockwise_edm_builders.at(link);
            ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                receiver_edm_builder.add_receiver_channel(
                    worker_receiver_semaphore_id,
                    // Since we are in worker signal EDM termination mode, we don't need to set the actual number of
                    // messages the EDM must forward as it will receive its finish signal from the worker instead
                    1,
                    receiver_worker_coords,
                    expected_message_size_bytes);
            edm_interface_addresses.worker_receiver_edm_semaphore_addresses.insert(
                {global_worker_idx, receiver_channel_buffer_info.eth_semaphore_l1_address});
            edm_interface_addresses.worker_receiver_edm_buffer_addresses.insert(
                {global_worker_idx, receiver_channel_buffer_info.eth_buffer_l1_address});
        }
    }
}

static std::tuple<KernelHandle, KernelHandle, KernelHandle, std::optional<KernelHandle>> build_reduce_scatter_worker_ct(
    tt::tt_metal::Program& program,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::CCLOpConfig const& op_config,
    ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    CoreRangeSet const& worker_core_range,
    // if line and at the end of the line we split the worker core range
    // because we need to invoke separate kernels
    std::optional<CoreRangeSet> const& split_worker_core_range,
    ttnn::operations::binary::BinaryOpType binary_math_op) {
    log_trace(tt::LogOp, "build_reduce_scatter_worker_ct");

    auto const& worker_defines = op_config.emit_worker_defines();
    TT_ASSERT(worker_defines.size() > 0);
    for (auto const& [key, value] : worker_defines) {
        log_trace(tt::LogOp, "Worker Define: {} = {}", key, value);
    }
    if (split_worker_core_range.has_value()) {
        log_trace(tt::LogOp, "second worker core list:");
        for (const auto &core : corerange_to_cores(split_worker_core_range.value())) {
            log_trace(tt::LogOp, "\tx={},y={}", core.x, core.y);
        }
    }

    static std::string const& receiver_kernel_path = //topology_config.is_linear ?
        // "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_line_reduce_scatter_reader.cpp" :
        "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_interleaved_ring_reduce_scatter_reader.cpp";
    static std::string const& forward_sender_kernel_path = "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/kernels/worker_interleaved_ring_reduce_scatter_sender.cpp";
    static std::string const& line_start_sender_kernel_path = "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send.cpp";
    static std::string const& reduce_kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp";

    // Need to be able to split up the workers so that on the end of the lines, some of the cores are for send/receive and
    // others are for CCL send only
    bool is_start_chip_in_line = topology_config.is_linear && (topology_config.ring_index == 0 || topology_config.ring_index == topology_config.ring_size - 1);

    // If we we implementing a line, and are at the end of the line
    bool worker_grid_split_in_half = is_start_chip_in_line;

    KernelHandle worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id;
    std::optional<KernelHandle> line_start_sender_kernel_id;

    worker_receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        receiver_kernel_path,
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_receiver_kernel_ct_args(), worker_defines));

    worker_sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        forward_sender_kernel_path,
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(worker_arg_builder.generate_sender_kernel_ct_args(), worker_defines));

    vector<uint32_t> compute_kernel_args = {};
    constexpr bool fp32_dest_acc_en = false;
    constexpr bool math_approx_mode = false;
    std::map<string, string> eltwise_defines = ttnn::operations::binary::utils::get_defines(binary_math_op);
    worker_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reduce_kernel_path,
        worker_core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = eltwise_defines});

    if (is_start_chip_in_line) {
        TT_ASSERT(split_worker_core_range.has_value(), "Internal Error. (line) Reduce scatter did not generate a smaller second worker grid to map the line start kernels onto");
        log_trace(tt::LogOp, "Invoking CCL send kernel on split kernel core range");
        for (auto const& core : corerange_to_cores(split_worker_core_range.value())) {
            log_trace(tt::LogOp, "\tcore=(x={},y={})", core.x, core.y);
        }
        line_start_sender_kernel_id = tt::tt_metal::CreateKernel(
            program,
            line_start_sender_kernel_path,
            split_worker_core_range.value(),
            tt::tt_metal::WriterDataMovementConfig(worker_arg_builder.generate_line_start_sender_kernel_ct_args(), worker_defines));
    }

    return {worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id, line_start_sender_kernel_id};
}

static void set_reduce_scatter_worker_rt(
    tt::tt_metal::Program& program,
    Device const* device,
    KernelHandle worker_receiver_kernel_id,
    KernelHandle worker_sender_kernel_id,
    KernelHandle worker_reduce_kernel_id,
    std::optional<KernelHandle> optional_line_start_ccl_send_kernel,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::CCLOpConfig const& op_config,
    ttnn::ccl::reduce_scatter_detail::ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& cw_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder>& ccw_edm_builders,
    EdmInterfaceAddresses const& edm_interface_addresses,
    WorkerAttributes &worker_attributes,
    std::size_t num_edm_channels,
    std::size_t edm_num_buffers_per_channel,
    ttnn::operations::binary::BinaryOpType binary_math_op) {
    bool is_in_clockwise_direction = worker_attributes.direction == Direction::CLOCKWISE;
    const std::size_t global_worker_index = get_global_worker_id(worker_attributes, num_edm_channels);

    if (!topology_config.is_first_device_in_line(is_in_clockwise_direction))
    {
        CoreCoord const& receiver_edm = is_in_clockwise_direction
                                            ? topology_config.eth_receiver_cores.at(worker_attributes.link)
                                            : topology_config.eth_sender_cores.at(worker_attributes.link);
        ttnn::ccl::WorkerXY receiver_edm_noc_coord = ttnn::ccl::WorkerXY(
            device->ethernet_core_from_logical_core(receiver_edm).x,
            device->ethernet_core_from_logical_core(receiver_edm).y);
        const uint32_t edm_core_semaphore_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index);

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_receiver_kernel_id,
            worker_attributes.location_logical,
            worker_arg_builder.generate_receiver_kernel_rt_args(
                receiver_edm_noc_coord, edm_core_semaphore_address, edm_core_buffer_address, worker_attributes));

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_reduce_kernel_id,
            worker_attributes.location_logical,
            worker_arg_builder.generate_reduce_op_kernel_rt_args(worker_attributes, topology_config.ring_size));
    }

    // if (!topology_config.is_last_device_in_line(is_in_clockwise_direction))
    {
        CoreCoord sender_edm = is_in_clockwise_direction ? topology_config.eth_sender_cores.at(worker_attributes.link)
                                                        : topology_config.eth_receiver_cores.at(worker_attributes.link);
        ttnn::ccl::WorkerXY const sender_edm_noc_coord = ttnn::ccl::WorkerXY(
            device->ethernet_core_from_logical_core(sender_edm).x, device->ethernet_core_from_logical_core(sender_edm).y);
        TT_ASSERT(sender_edm_noc_coord.y == 0 || sender_edm_noc_coord.y == 6);
        const uint32_t edm_core_semaphore_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index);
        WorkerEdmInterfaceArgs edm_interface = {
            sender_edm_noc_coord.x,
            sender_edm_noc_coord.y,
            edm_core_buffer_address,
            edm_core_semaphore_address,
            edm_num_buffers_per_channel};

        log_trace(tt::LogOp, "hh7");
        bool use_line_start_kernel = topology_config.is_first_device_in_line(is_in_clockwise_direction);

        auto const rt_args = use_line_start_kernel
                                 ? worker_arg_builder.generate_line_start_sender_kernel_rt_args(
                                       edm_interface, worker_arg_builder.scatter_dim, worker_attributes)
                                 : worker_arg_builder.generate_sender_kernel_rt_args(edm_interface, worker_attributes);
        TT_ASSERT(!use_line_start_kernel || optional_line_start_ccl_send_kernel.has_value());
        auto sender_kernel_id = use_line_start_kernel ? optional_line_start_ccl_send_kernel.value(): worker_sender_kernel_id;

        tt::tt_metal::SetRuntimeArgs(
            program,
            sender_kernel_id,
            worker_attributes.location_logical,
            rt_args);
    }
}

/*
 * Core range sets for line topology
 */
static std::pair<CoreRangeSet, std::optional<CoreRangeSet>> select_worker_cores_for_line_topology(ttnn::ccl::RingTopology const& topology_config, ttnn::ccl::CCLOpConfig const& op_config, std::size_t num_links, std::size_t num_edm_channels) {
    static constexpr std::size_t num_directions_per_line = 2;

    TT_ASSERT(num_edm_channels % 2 == 0, "For line topologies, we expect a multiple of 2 number of channels for the algorithm and worker kernels to work.");
    const std::size_t workers_per_direction = num_edm_channels / num_directions_per_line;
    auto const& lower_half_of_cores = CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(workers_per_direction - 1, num_links - 1))});
    auto const& upper_half_of_cores = CoreRangeSet({CoreRange(CoreCoord(workers_per_direction, 0), CoreCoord(num_edm_channels - 1, num_links - 1))});
    if (topology_config.ring_index == 0) {
        log_trace(tt::LogOp, "Start of line, putting CCL send cores in lower half");
        return {upper_half_of_cores, lower_half_of_cores};
    } else if (topology_config.ring_index == topology_config.ring_size - 1) {
        // Flip them for the other end because the send will be for the "second" core range set (conceptually, the other direction)
        // of the line flows in the second half of all workers, for each chip.
        log_trace(tt::LogOp, "End of line, putting CCL send cores in lower half");
        return {lower_half_of_cores, upper_half_of_cores};
    } else {
        log_trace(tt::LogOp, "Middle of line - no CCL kernel");
        return {CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(num_edm_channels - 1, num_links - 1))}), std::nullopt};
    }
}

/*
 * Returns 1 or 2 core range sets. Typically returns only one but in the case of a line reduce scatter where we are at the end of the line,
 * then we must split the core range in half (and return 2), one for each direction where half the cores will invoke the ccl::send kernel
 * to implement the start of the line and the others will invoke the typical reduce scatter worker kernels.
 */
static std::pair<CoreRangeSet, std::optional<CoreRangeSet>> select_worker_cores(
    ttnn::ccl::RingTopology const& topology_config, ttnn::ccl::CCLOpConfig const& op_config, std::size_t num_links, std::size_t num_edm_channels) {
    switch (op_config.get_topology()) {
        case ttnn::ccl::Topology::Linear: {
            auto const& core_ranges = select_worker_cores_for_line_topology(topology_config, op_config, num_links, num_edm_channels);
            log_trace(tt::LogOp, "First core range");
            for (const auto &core : corerange_to_cores(core_ranges.first)) {
                log_trace(tt::LogOp, "\tx={},y={}", core.x, core.y);
            }
            if (core_ranges.second.has_value()) {
                log_trace(tt::LogOp, "second worker core list:");
                for (const auto &core : corerange_to_cores(core_ranges.second.value())) {
                    log_trace(tt::LogOp, "\tx={},y={}", core.x, core.y);
                }
            }
            return core_ranges;
        }

        case ttnn::ccl::Topology::Ring:
            return {CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(num_edm_channels - 1, num_links - 1))}), std::nullopt};

        default: TT_ASSERT(false, "Unsupported topology"); return {CoreRangeSet({}), std::nullopt};
    };
}

static WorkerTransferInfo compute_num_edm_messages_per_channel(
    ccl::CCLOpConfig const& op_config,
    RingReduceScatterWrappedTensorSlicer& tensor_slicer,  // TODO: Update to Generic ReduceScatterSlicer when it is implemented
    std::vector<ttnn::ccl::EriscDatamoverBuilder> const& cw_per_link_edm_builders,
    std::vector<ttnn::ccl::EriscDatamoverBuilder> const& ccw_per_link_edm_builders,
    std::size_t const num_edm_channels,
    std::size_t const num_links,
    std::size_t const ring_size) {
    uint32_t const page_size_in_bytes = op_config.get_page_size();
    TT_ASSERT(num_edm_channels > 0);
    TT_ASSERT(num_links > 0);
    TT_ASSERT(page_size_in_bytes > 0);
    log_trace(tt::LogOp, "WorkerTransferInfo");
    std::size_t total_num_workers = num_edm_channels * num_links;

    auto get_iter_begin = [num_edm_channels](auto& vec, std::size_t link) -> auto {
        return vec.begin() + (link * num_edm_channels);
    };

    auto get_iter_end = [num_edm_channels, num_links](auto& vec, std::size_t link) -> auto {
        bool last_link = link == num_links - 1;
        TT_ASSERT(
            (!last_link && ((link + 1) * num_edm_channels < vec.size())) ||
            (last_link && ((link + 1) * num_edm_channels == vec.size())));
        return last_link ? vec.end() : vec.begin() + ((link + 1) * num_edm_channels);
    };

    // Pages per EDM channel
    std::size_t total_num_edm_channels = num_links * num_edm_channels;
    log_trace(tt::LogOp, "total_num_edm_channels: {}", total_num_edm_channels);

    std::vector<uint32_t> num_pages_per_full_chunk(total_num_edm_channels * num_links, 0);

    for (std::size_t link = 0; link < num_links; link++) {
        std::size_t edm_channel_size_in_bytes = cw_per_link_edm_builders.at(link).get_eth_buffer_size_bytes();
        std::size_t num_pages_per_edm_buffer = edm_channel_size_in_bytes / page_size_in_bytes;
        log_trace(
            tt::LogOp,
            "link {}, edm_channel_size_in_bytes: {}, page_size_in_bytes: {}, num_pages_per_edm_buffer: {}",
            link,
            edm_channel_size_in_bytes,
            page_size_in_bytes,
            num_pages_per_edm_buffer);

        std::fill(
            get_iter_begin(num_pages_per_full_chunk, link),
            get_iter_end(num_pages_per_full_chunk, link),
            num_pages_per_edm_buffer);
    }

    log_trace(tt::LogOp, "-- num_pages_per_full_chunk:");
    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t w = 0; w < num_edm_channels; w++) {
            log_trace(
                tt::LogOp, "\t\t(link={},worker={}): {}", l, w, num_pages_per_full_chunk.at(l * num_edm_channels + w));
        }
    }

    return WorkerTransferInfo(num_pages_per_full_chunk, num_links, num_edm_channels);
}

static uint32_t compute_maximum_worker_slice_in_bytes(
    ttnn::ccl::Topology topology,
    uint32_t cb_src0_size_pages,
    uint32_t cb_dst0_size_pages,
    uint32_t cb_short_circuit_size_pages,
    std::size_t edm_channel_buffer_size,
    uint32_t page_size) {
    switch (topology) {
        case ttnn::ccl::Topology::Linear:
            // For linear topology, we only want one slice per worker so we don't
            return std::numeric_limits<uint32_t>::max();

        case ttnn::ccl::Topology::Ring:
            return std::min(cb_short_circuit_size_pages, cb_src0_size_pages + cb_dst0_size_pages) * page_size +
                   edm_channel_buffer_size;

        default: TT_ASSERT(false, "Unsupported topology"); return 0;
    };
}

static bool is_cb_buffering_sufficient_to_avoid_deadlock(
    ttnn::ccl::Topology topology,
   ttnn::ccl::InterleavedTensorWorkerSlice const& worker_slice,
    uint32_t cb_src0_size_pages,
    uint32_t cb_dst0_size_pages,
    uint32_t cb_short_circuit_size_pages,
    std::size_t edm_channel_buffer_size,
    uint32_t page_size) {
    uint32_t worker_size_pages_rounded_up =
        tt::round_up(worker_slice.worker_slice_shape.x * worker_slice.worker_slice_shape.y, cb_src0_size_pages / 2);
    uint32_t worker_slice_size_bytes = worker_size_pages_rounded_up * page_size;
    uint32_t available_buffering_capacity = compute_maximum_worker_slice_in_bytes(
        topology, cb_src0_size_pages, cb_dst0_size_pages, cb_short_circuit_size_pages, edm_channel_buffer_size, page_size);
    log_trace(tt::LogOp, "worker_slice.worker_slice_shape.x: {}", worker_slice.worker_slice_shape.x);
    log_trace(tt::LogOp, "worker_slice.worker_slice_shape.y: {}", worker_slice.worker_slice_shape.y);
    log_trace(tt::LogOp, "worker_slice_size_bytes: {}", worker_slice_size_bytes);
    log_trace(tt::LogOp, "worker_size_pages_rounded_up: {}", worker_size_pages_rounded_up);
    log_trace(tt::LogOp, "cb_src0_size_pages: {}", cb_src0_size_pages);
    log_trace(tt::LogOp, "cb_dst0_size_pages: {}", cb_dst0_size_pages);
    log_trace(tt::LogOp, "page_size: {}", page_size);
    log_trace(tt::LogOp, "edm_channel_buffer_size: {}", edm_channel_buffer_size);
    log_trace(tt::LogOp, "available_buffering_capacity: {}", available_buffering_capacity);

    return available_buffering_capacity >= worker_slice_size_bytes;
}

static std::tuple<CBHandle, CBHandle, CBHandle, CBHandle> create_worker_circular_buffers(
    Tensor const& input_tensor,
   ttnn::ccl::CCLOpConfig const& op_config,
    CoreRangeSet const& worker_core_range,
    uint32_t worker_pages_per_transfer,
    tt::tt_metal::Program& program) {
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t page_size_bytes = op_config.get_page_size();

    // Input 0 CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // Input 1 CB
    uint32_t src1_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{src1_cb_index, df}})
            .set_page_size(src1_cb_index, page_size_bytes);
    CBHandle cb_src1_workers = CreateCircularBuffer(program, worker_core_range, cb_src1_config);

    // Dataflow Writer Kernel input CB
    uint32_t cb_dst0_index = tt::CB::c_out0;
    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(worker_pages_per_transfer * page_size_bytes, {{cb_dst0_index, df}})
            .set_page_size(cb_dst0_index, page_size_bytes);
    CBHandle cb_dst0_sender_workers = CreateCircularBuffer(program, worker_core_range, cb_dst0_config);

    // From reader -> writer kernel (I think I need this because sharing the cb_dst0_sender_workers as output
    // of reader kernel (first output) and math kernel (all subsequent outputs) doesn't seem to work because
    // it seems like the math kernels hold some of the CB state in local variables)
    uint32_t cb_short_circuit_index = tt::CB::c_out1;
    tt::tt_metal::CircularBufferConfig cb_short_circuit_config =
        tt::tt_metal::CircularBufferConfig(
            (worker_pages_per_transfer * page_size_bytes) * 2, {{cb_short_circuit_index, df}})
            .set_page_size(cb_short_circuit_index, page_size_bytes);
    CBHandle cb_short_circuit_sender_workers =
        CreateCircularBuffer(program, worker_core_range, cb_short_circuit_config);

    return {cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers, cb_short_circuit_sender_workers};
}

static std::vector<WorkerAttributes> build_worker_attributes(
    ttnn::ccl::RingTopology const& topology_config,
    std::vector<CoreCoord> const& worker_cores_list,
    std::optional<std::vector<CoreCoord>> const& second_worker_cores_list,

    std::size_t num_links,
    std::size_t num_channels_per_link,
    std::function<bool(std::size_t)> is_buffer_in_clockwise_direction_fn) {

    std::vector<WorkerAttributes> worker_attributes;

    std::size_t workers_per_direction_per_link = num_channels_per_link / (topology_config.is_linear ? 2 : 1);

    std::size_t worker_cores_idx = 0;
    std::size_t second_worker_cores_idx = 0;

    bool split_grids = second_worker_cores_list.has_value();
    auto const first_workers_list = split_grids && topology_config.ring_index == 0 ?
        second_worker_cores_list.value():
        worker_cores_list;

    std::optional<std::vector<CoreCoord>> second_workers_list =
        !topology_config.is_linear ? first_workers_list :
        !split_grids ? first_workers_list :
        topology_config.ring_index == 0 ? first_workers_list :
        second_worker_cores_list.has_value() ? second_worker_cores_list.value() : std::optional<std::vector<CoreCoord>>(std::nullopt);

    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t i = 0; i < workers_per_direction_per_link; i++) {
            auto worker_id = get_global_worker_id(l, i, num_channels_per_link);
            TT_ASSERT(worker_cores_idx < worker_cores_list.size());

            worker_attributes.push_back(
                {
                    l,
                    i,
                    is_buffer_in_clockwise_direction_fn(worker_id) ? Direction::CLOCKWISE : Direction::COUNTER_CLOCKWISE,
                    first_workers_list[worker_cores_idx]
                }
            );
            worker_cores_idx++;
        }
        if (topology_config.is_linear) {
            auto & second_vec_index = split_grids ? second_worker_cores_idx : worker_cores_idx;
            for (std::size_t i = 0; i < workers_per_direction_per_link; i++) {
                TT_ASSERT(second_vec_index < second_workers_list.value().size());
                std::size_t my_logical_index = workers_per_direction_per_link + i;
                worker_attributes.push_back(
                    {
                        l,
                        my_logical_index,
                        is_buffer_in_clockwise_direction_fn(my_logical_index) ?
                            Direction::CLOCKWISE : Direction::COUNTER_CLOCKWISE,
                        second_workers_list.value()[second_vec_index],
                    }
                );
                std::size_t my_idx = worker_attributes.size() - 1;
                std::size_t associated_idx = my_idx - workers_per_direction_per_link;
                worker_attributes[my_idx].associated_worker_index = associated_idx;
                worker_attributes[my_idx].associated_worker_core_logical = worker_attributes[associated_idx].location_logical;
                worker_attributes[associated_idx].associated_worker_index = my_idx;
                worker_attributes[associated_idx].associated_worker_core_logical = worker_attributes[my_idx].location_logical;
                second_vec_index++;
            }
        }
    }

    log_trace(tt::LogOp, "Worker Attributes:");
    for (const auto &wa : worker_attributes) {
        log_trace(tt::LogOp, "\tAttributes: link={}, index={}, core_logical=(x={},y={}), direction={}, associated_core=(x={},y={}), associated_index={}",
            wa.link,
            wa.channel,
            wa.location_logical.x,
            wa.location_logical.y,
            wa.direction == Direction::CLOCKWISE ? "CLOCKWISE": "COUNTER-CLOCKWISE",
            wa.associated_worker_core_logical.has_value() ? std::to_string(wa.associated_worker_core_logical.value().x) : "std::nullopt",
            wa.associated_worker_core_logical.has_value() ? std::to_string(wa.associated_worker_core_logical.value().y) : "std::nullopt",
            wa.associated_worker_index.has_value() ? std::to_string(wa.associated_worker_index.value()) : "std::nullopt"
            );

    }

    TT_ASSERT(!topology_config.is_linear || std::ranges::all_of(worker_attributes, [](auto const& wa) { return wa.associated_worker_index.has_value() && wa.associated_worker_core_logical.has_value(); }));

    return worker_attributes;
}

operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    ttnn::operations::binary::BinaryOpType reduce_op,
    const uint32_t scatter_split_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
   ttnn::ccl::Topology topology) {

    // if (ring_index != 0) {
        // sleep for 10s * ring_index for debug purposes to serialize the log
        std::this_thread::sleep_for(std::chrono::seconds(10 * ring_index));
    // }

    log_trace(tt::LogOp, "reduce_scatter_with_workers entry");
    TT_ASSERT(
        input_tensor.get_legacy_shape()[scatter_split_dim] ==
            output_tensor.get_legacy_shape()[scatter_split_dim] * ring_size,
        "Input and output tensor shapes must match");
    TT_ASSERT(
        input_tensor.buffer()->num_pages() % ring_size == 0,
        "Reduce scatter current only supports even divisibility of input tensor(s) across ranks");

    /////////////// Constants/Configuration
    /// Constants/Configuration
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode =ttnn::ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    auto const& op_config =ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    std::unique_ptr<ttnn::ccl::CclOpTensorConfig> input_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(input_tensor);
    std::unique_ptr<ttnn::ccl::CclOpTensorConfig> output_tensor_config =
        ttnn::ccl::CclOpTensorConfig::build_all_gather_tensor_config(output_tensor);
    // // The input tensor is fractured by ring_size so we divi
    std::size_t input_tensor_n_elems_per_slice = input_tensor.volume() / ring_size;
    uint32_t input_tensor_num_units_per_tensor_slice =
        input_tensor_n_elems_per_slice / (tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT);

    TT_ASSERT(input_tensor_num_units_per_tensor_slice > 0);
    uint32_t max_num_workers = std::min<std::size_t>(8, input_tensor_num_units_per_tensor_slice);
    bool enable_bidirectional = true;
    auto num_edm_channels_per_link = decide_number_of_edm_channels(op_config, max_num_workers, enable_bidirectional);
    log_trace(tt::LogOp, "num_edm_channels_per_link: {}", num_edm_channels_per_link);
    auto edm_termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    constexpr std::size_t num_buffers_per_channel = 1; // enable double buffering later
    auto const& edm_builder = create_erisc_datamover_builder(
        num_edm_channels_per_link, op_config.get_page_size(), num_buffers_per_channel, buffer_sharing_mode, edm_termination_mode);
    TT_ASSERT(num_edm_channels_per_link > 0);

    Tensor const& local_chip_tensor = input_tensor;
    Tensor const& local_chip_output_tensor = output_tensor;

    std::map<string, string> worker_defines;
    std::vector<ttnn::ccl::EriscDatamoverBuilder> cw_per_link_edm_builders(num_links, edm_builder);
    std::vector<ttnn::ccl::EriscDatamoverBuilder> ccw_per_link_edm_builders(num_links, edm_builder);

    const auto& device = local_chip_tensor.device();
    auto const& topology_config =
       ttnn::ccl::RingTopology(device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);
    bool is_linear = topology_config.is_linear;
    std::function<bool(uint32_t)> is_worker_in_clockwise_direction_fn = [is_linear, enable_bidirectional, num_edm_channels_per_link](uint32_t x) {
                static constexpr bool bidirectional_directions = 2;
                return is_linear ? (x < (num_edm_channels_per_link / bidirectional_directions)):
                    enable_bidirectional ? (x % bidirectional_directions == 0) : true;
            };

    auto const& [worker_core_range, second_worker_core_range] = select_worker_cores(topology_config, op_config, num_links, num_edm_channels_per_link);
    auto const& worker_cores = corerange_to_cores(worker_core_range, std::nullopt, true);
    std::optional<std::vector<CoreCoord>> second_worker_cores_list;
    if (second_worker_core_range.has_value()) {
        second_worker_cores_list = corerange_to_cores(second_worker_core_range.value(), std::nullopt, true);
    }
    std::vector<WorkerAttributes> all_worker_attributes = build_worker_attributes(
        topology_config,
        worker_cores,
        second_worker_cores_list,
        num_links,
        num_edm_channels_per_link,
        is_worker_in_clockwise_direction_fn);
    //////////////////
    tt::tt_metal::Program program{};
    // Issue #10978: CCLs need to be tagged as having multi-device dependencies, when running on Galaxy.
    program.capture_multi_device_dependencies();

    // Semaphores && CBs
    auto worker_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
    auto worker_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
    std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id = std::nullopt;

    if (topology_config.is_linear) {
        receiver_worker_partial_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range, 0);
    }

    uint32_t cb_num_pages = std::min(input_tensor_num_units_per_tensor_slice,
        (cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes() / op_config.get_page_size())) * 2;
    uint32_t cb_num_pages_per_packet = cb_num_pages / 2;
    log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
    auto const& [cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers, cb_short_circuit_sender_workers] =
        create_worker_circular_buffers(local_chip_tensor, op_config, worker_core_range, cb_num_pages, program);



    uint32_t max_worker_slice_in_bytes = compute_maximum_worker_slice_in_bytes(
        topology,
        cb_num_pages,
        cb_num_pages,
        cb_num_pages,
        cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes(),
        op_config.get_page_size());
    std::size_t num_workers = all_worker_attributes.size();
    TT_ASSERT(num_workers == num_edm_channels_per_link * num_links);
    auto tensor_slicer = ttnn::ccl::RingReduceScatterWrappedTensorSlicer(
        local_chip_tensor,
        local_chip_output_tensor,
        scatter_split_dim,
        ring_index,
        ring_size,
        num_workers,
        max_worker_slice_in_bytes,
        cb_num_pages / 2);

    // Not per buffer because the buffer sharing mode may cause some buffers to share EDM transfers
    WorkerTransferInfo const& worker_transfer_info = compute_num_edm_messages_per_channel(
        op_config,
        tensor_slicer,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        num_edm_channels_per_link,
        num_links,
        ring_size);

    // Configure the EDM builders
    EdmInterfaceAddresses edm_interface_addresses;
    for (std::size_t link = 0; link < num_links; link++) {
        add_worker_config_to_edm_builders(
            device,
            tensor_slicer,
            op_config,
            all_worker_attributes,
            num_edm_channels_per_link,
            num_buffers_per_channel,

            cw_per_link_edm_builders,
            ccw_per_link_edm_builders,

            worker_sender_semaphore_id,
            worker_receiver_semaphore_id,
            link,
            ring_size, // Replace with TopologyConfig
            ring_index,

            edm_interface_addresses);
    }

    // build worker kernels ct
    auto const& dummy_worker_slice = tensor_slicer.get_worker_slice(0);
    auto worker_arg_builder = ReduceScatterWorkerArgBuilder(
        device,
        op_config,
        topology_config,
        dummy_worker_slice,
        worker_transfer_info,
        edm_termination_mode, // Can probably remove this once everything is working
        scatter_split_dim,
        cb_num_pages_per_packet,
        worker_sender_semaphore_id,
        worker_receiver_semaphore_id,
        receiver_worker_partial_ready_semaphore_id,
        num_buffers_per_channel);
    auto [worker_receiver_kernel_id, worker_sender_kernel_id, worker_reduce_kernel_id, optional_line_start_ccl_send_kernel] = build_reduce_scatter_worker_ct(
        program,
        topology_config,
        op_config,
        worker_arg_builder,
        worker_core_range,
        second_worker_core_range,
        reduce_op);



    // build the worker kernels
    // std::size_t num_duplicate_directions = topology_config.is_linear ? 2 : 1;
    // for (std::size_t direction = 0; direction < num_duplicate_directions; direction++) {

        // set worker kernels rt
        tt::tt_metal::ComputeConfig compute_config;
        for (std::size_t link = 0; link < num_links; link++) {
            log_trace(tt::LogOp, "==============================================");
            log_trace(tt::LogOp, "------------------ Link: {} ------------------", link);
            for (std::size_t worker = 0; worker < num_edm_channels_per_link; worker++) {
                std::size_t global_worker_index = worker + link * num_edm_channels_per_link;

                log_trace(tt::LogOp, "------ Worker: {} (global ID={})", worker, global_worker_index);

                    std::size_t worker_tensor_slice_index = !topology_config.is_linear ?
                        global_worker_index :
                        (worker % num_edm_channels_per_link / 2) + (num_edm_channels_per_link / 2) * link;
                    auto const& worker_slice = tensor_slicer.get_worker_slice(worker_tensor_slice_index);
                    log_trace(tt::LogOp, "here");
                auto worker_arg_builder = ReduceScatterWorkerArgBuilder(
                    device,
                    op_config,
                    topology_config,
                    worker_slice,
                    worker_transfer_info,
                    edm_termination_mode,
                    scatter_split_dim,
                    cb_num_pages_per_packet,
                    worker_sender_semaphore_id,
                    worker_receiver_semaphore_id,
                    receiver_worker_partial_ready_semaphore_id,
                    num_buffers_per_channel);

                // log_trace(tt::LogOp, "worker_cores.at(global_worker_index): {}", worker_cores.at(global_worker_index));
                set_reduce_scatter_worker_rt(
                    program,
                    device,
                    worker_receiver_kernel_id,
                    worker_sender_kernel_id,
                    worker_reduce_kernel_id,
                    optional_line_start_ccl_send_kernel,
                    topology_config,
                    op_config,
                    worker_arg_builder,
                    cw_per_link_edm_builders,
                    ccw_per_link_edm_builders,
                    edm_interface_addresses,
                    all_worker_attributes.at(global_worker_index),
                    num_edm_channels_per_link,
                    num_buffers_per_channel,
                    reduce_op);

                TT_FATAL(is_cb_buffering_sufficient_to_avoid_deadlock(
                        topology,
                        worker_slice,
                        cb_num_pages,
                        cb_num_pages,
                        cb_num_pages,
                        cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes(),
                        op_config.get_page_size()), "Internal error: reduce scatter implementation generated a program that will deadlock due to insufficient buffering based on the tensor slice sizes the op chose to use.");
            }
        }
    // }

    // Generate the EDM kernels
   ttnn::ccl::generate_edm_kernels_for_ring_or_linear_topology(
        program,
        device,
        topology_config,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        receiver_device_id,
        sender_device_id);

    std::size_t total_num_workers = worker_cores.size();
    auto override_runtime_arguments_callback =
        [topology_config, worker_receiver_kernel_id, worker_sender_kernel_id, worker_cores, total_num_workers, ring_index](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors.at(0);
            const auto& output = output_tensors.at(0);
            auto &worker_receiver_runtime_args_by_core = GetRuntimeArgs(program, worker_receiver_kernel_id);
            auto &worker_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_kernel_id);
            for (uint32_t i = 0; i < worker_cores.size(); ++i) {
                auto core = worker_cores.at(i);
                auto& worker_receiver_runtime_args = worker_receiver_runtime_args_by_core[core.x][core.y];
                worker_receiver_runtime_args.at(0) = input.buffer()->address();

                auto& worker_sender_runtime_args = worker_sender_runtime_args_by_core[core.x][core.y];
                worker_sender_runtime_args.at(0) = output.buffer()->address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace reduce_scatter_detail
}  // namespace ccl
}  // namespace ttnn
