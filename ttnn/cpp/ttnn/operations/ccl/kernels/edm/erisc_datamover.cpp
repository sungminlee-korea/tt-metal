// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "eth_l1_address_map.h"

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_async_datamover.hpp"

// Args Schema:
// 1) handshake addr
// 2) sender channels offset (indicates for the erisc channels, where the senders start
//    so sender and receivers don't clash when paired with sender/receiver on the other
//    end of the link.)
// 3) sender num channels (How many erisc channels to use. ALso how many buffers to instantiate)
//    Informs how many times to iterate through the next group of args
//    4) sender_buffer_address
//    5) sender_num_messages_to_send
//    6) sender_channel_size
//    7) sender_semaphores_base_address
//    8) worker_semaphore_id
//    9) sender_num_workers
//       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
//       10) worker_coord(s)
// ...
// Repeat from step 2 for receiver side

using ttnn::ccl::WorkerXY;

// struct sender_receiver_index_t {
//     static constexpr uint16_t COMBINED_INCR = (1 << 8) | 1;
//     union {
//         struct {
//             uint8_t sender;
//             uint8_t receiver;
//         };
//         uint16_t combined;
//     } index;
//     union {
//         struct {
//             uint8_t sender;
//             uint8_t receiver;
//         };
//         uint16_t combined;
//     } real_index;
//     union {
//         struct {
//             uint8_t sender;
//             uint8_t receiver;
//         };
//         uint16_t combined;
//     } start;

//     uint8_t num_senders;
//     uint8_t num_receivers;

//     sender_receiver_index_t(uint8_t send_start, uint8_t receive_start, uint8_t num_senders, uint8_t num_receivers) {
//         start.sender = send_start;
//         start.receiver = receive_start;
//         index.sender = 0;
//         index.receiver = 0;
//         real_index.sender = send_start;
//         real_index.receiver = receive_start;
//         num_senders = num_senders;
//         num_receivers = num_receivers;
//     }

//     FORCE_INLINE void increment() {
//         index.sender += 1;
//         index.receiver += 1;
//         index.sender = index.sender >= num_senders ? 0 : index.sender;
//         index.receiver = index.receiver >= num_receivers ? 0 : index.receiver;
//         real_index.combined = start.combined + index.combined;
//     }
// };


template <typename T>
struct channel_counter_t {
    channel_counter_t(T offset, T num) :
        last_val(offset + num - 1), start_offset(offset), curr(start_offset) {
        }

    FORCE_INLINE void increment() {
        bool last = (curr == last_val);
        curr = last ? start_offset : curr + 1;
    }

    FORCE_INLINE T get() const { return curr; }

    protected:
    T last_val;
    T start_offset;
    T curr;
};

void kernel_main() {
    static constexpr ttnn::ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode =
        static_cast<ttnn::ccl::EriscDataMoverBufferSharingMode>(get_compile_time_arg_val(0));

    static constexpr ttnn::ccl::EriscDataMoverTerminationMode terminate_on_worker_signal =
        static_cast<ttnn::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(1));

    static constexpr uint32_t chip_id = get_compile_time_arg_val(2);

    using EDM_CONFIG_T = erisc::datamover::
        EriscDatamoverConfig<edm_buffer_sharing_mode, terminate_on_worker_signal>;
    using ChannelBufferT = erisc::datamover::ChannelBuffer<EDM_CONFIG_T>;

    std::array<ChannelBufferT, ttnn::ccl::edm::MAX_NUM_CHANNELS_ALLOWED> buffer_channels;

    // SENDER ARGS
    uint32_t args_offset = 0;
    const bool is_handshake_master = get_arg_val<uint32_t>(args_offset++) != 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(args_offset++);

    bool is_done_as_rx_handshaker = is_handshake_master;
    if (is_handshake_master) {
        erisc::datamover::handshake::sender_side_start(handshake_addr);
    } else {
        erisc::datamover::handshake::receiver_side_start(handshake_addr);
    }

    const uint32_t sender_num_channels = get_arg_val<uint32_t>(args_offset++);
    const uint32_t receiver_num_channels = get_arg_val<uint32_t>(args_offset++);
    const uint32_t num_buffers_per_channel = get_arg_val<uint32_t>(args_offset++);
    ASSERT(ttnn::ccl::edm::MAX_NUM_CHANNELS_ALLOWED >= sender_num_channels + receiver_num_channels);
    ASSERT(num_buffers_per_channel > 0);

    // Receiver args
    uint8_t const receiver_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint8_t num_receivers_with_no_work = 0;
    for (uint32_t channel = 0; channel < receiver_num_channels; channel++) {
        uint32_t const receiver_buffers_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const receiver_channel_size = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const worker_semaphore_address = get_semaphore(get_arg_val<uint32_t>(args_offset++));
        uint32_t const receiver_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += receiver_num_workers;
        new (&buffer_channels[receiver_channels_start + channel]) ChannelBufferT(
            receiver_channels_start + channel,
            receiver_buffers_base_address,
            receiver_channel_size,
            worker_semaphore_address,
            receiver_num_workers,
            receiver_num_messages_to_send,
            num_buffers_per_channel,
            (volatile tt_l1_ptr uint32_t *const)receiver_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            false);

        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (receiver_num_messages_to_send == 0) {
                num_receivers_with_no_work++;
            }
        }
    }

    if (!is_handshake_master) {
        if (!is_done_as_rx_handshaker && erisc::datamover::handshake::receiver_side_can_finish()) {
            is_done_as_rx_handshaker = true;
            erisc::datamover::handshake::receiver_side_finish(handshake_addr);
        }
    }

    uint8_t const sender_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint8_t num_senders_with_no_work = 0;
    for (uint32_t channel = 0; channel < sender_num_channels; channel++) {
        uint32_t const sender_buffer_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const sender_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const sender_channel_size = get_arg_val<uint32_t>(args_offset++);
        // The erisc's local l1 copy of the semaphore workers remotely increment
        uint32_t const sender_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        // worker's semaphore L1 address
        const uint32_t worker_semaphore_address = get_semaphore(get_arg_val<uint32_t>(args_offset++));
        const uint32_t sender_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += sender_num_workers;
        new (&buffer_channels[sender_channels_start + channel]) ChannelBufferT(
            sender_channels_start + channel,
            sender_buffer_address,
            sender_channel_size,
            worker_semaphore_address,
            sender_num_workers,
            sender_num_messages_to_send,
            num_buffers_per_channel,
            (volatile tt_l1_ptr uint32_t *const)sender_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            true);
        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (sender_num_messages_to_send == 0) {
                num_senders_with_no_work++;
            }
        }
    }

    if (is_handshake_master) {
        erisc::datamover::handshake::sender_side_finish(handshake_addr);
    } else {
        if (!is_done_as_rx_handshaker) {
            erisc::datamover::handshake::receiver_side_finish(handshake_addr);
            is_done_as_rx_handshaker = true;
        }
    }
    uint32_t eth_transaction_ack_word_addr = handshake_addr + 16;
    uint32_t eth_transaction_complete_addr = handshake_addr + 32;

    constexpr uint32_t SWITCH_INTERVAL = 4000000;
    uint32_t did_nothing_count = 0;

    uint32_t num_senders_complete = num_senders_with_no_work;
    uint32_t num_receivers_complete = num_receivers_with_no_work;
    bool senders_in_progress = num_senders_complete != sender_num_channels;
    bool receivers_in_progress = num_receivers_complete != receiver_num_channels;

    channel_counter_t sender_index = channel_counter_t<uint16_t>{sender_channels_start, static_cast<uint16_t>(sender_num_channels)};
    channel_counter_t receiver_index = channel_counter_t<uint16_t>{receiver_channels_start, static_cast<uint16_t>(receiver_num_channels)};

    while (senders_in_progress || receivers_in_progress) {
        bool did_something_sender = false;
        bool did_something_receiver = false;

        uint32_t num_receivers_complete_old = num_receivers_complete;
        uint32_t num_senders_complete_old = num_senders_complete;
        //////////////////////////////////////
        // SENDER
        if (senders_in_progress) {
            ChannelBufferT &current_sender = buffer_channels[sender_index.get()];
            switch (current_sender.get_state()) {
                case ChannelBufferT::STATE::SENDER_WAITING_FOR_WORKER:
                did_something_sender =
                    erisc::datamover::sender_noc_receive_payload_ack_check_sequence(current_sender, num_senders_complete);
                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                break;

                case ChannelBufferT::STATE::SENDER_READY_FOR_ETH_TRANSFER:
                did_something_sender = erisc::datamover::sender_eth_send_data_sequence(current_sender);
                    break;

                case ChannelBufferT::STATE::SENDER_SIGNALING_WORKER:
                did_something_sender = erisc::datamover::sender_notify_workers_if_buffer_available_sequence(
                                    current_sender, num_senders_complete);
                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                break;

                case ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH:
                did_something_sender =
                    erisc::datamover::sender_eth_check_receiver_ack_sequence(current_sender, num_senders_complete);
                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                break;

                default:
                break;
            };
        }

        if (receivers_in_progress) {
            ChannelBufferT &current_receiver = buffer_channels[receiver_index.get()];
            //////////////////////////////////////
            // RECEIVER

            switch (current_receiver.get_state()) {
                case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_ETH:
                did_something_receiver = erisc::datamover::receiver_eth_accept_payload_sequence(current_receiver, num_receivers_complete, eth_transaction_ack_word_addr);
                receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
                break;

                case ChannelBufferT::STATE::RECEIVER_SIGNALING_WORKER:
                did_something_receiver =
                    erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(current_receiver);
                break;

                case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_WORKER:
                did_something_receiver = erisc::datamover::receiver_noc_read_worker_completion_check_sequence(
                                    current_receiver, num_receivers_complete);
                receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
                break;

                default:
                break;
            };
        }
        sender_index.increment();
        receiver_index.increment();

        //////////////////////////////////////

        // Enabling this block as is (with all the "did_something"s, seems to cause a loss of about
        // 0.5 GBps in throughput)
        if (did_something_sender || did_something_receiver) {
            did_nothing_count = 0;
        } else {
            if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                run_routing();
            }
        }
    }

    {
        for (uint32_t s = 0; s < sender_num_channels + receiver_num_channels; s++) {
            auto &channel = buffer_channels[s];
            // We need to explicitly check for channel send done because we may
            // advance sender channel state as soon as we receive an ack. Since we
            // may be the last active channel, and advance to done state just from ack
            // from the receiver ("I got a payload"), then we need to wait for done
            // at the very end here. Otherise if we invoke another erisc op back-to-back,
            // we may mess up transaction state because it's possible for receiver of this
            // op to send the completion done after that one has already started.
            uint32_t wait_count = 0;
            uint32_t wait_max = 5000000;
            for (uint8_t buffer_index = 0; buffer_index < num_buffers_per_channel; buffer_index++) {
                wait_count = 0;
                channel.buffer_index = buffer_index;
                if (!channel.is_sender_side) {
                    if (!channel.eth_is_receiver_channel_send_done()) {
                        channel.eth_receiver_channel_done();
                    }
                }
            }
            for (uint8_t buffer_index = 0; buffer_index < num_buffers_per_channel; buffer_index++) {
                if (channel.is_sender_side) {
                    while (!channel.eth_is_receiver_channel_send_done()) {
                        wait_count++;
                        if (wait_count > wait_max) {
                            DEBUG_STATUS("STK");
                            run_routing();
                            wait_count = 0;
                        }
                    }
                }
            }
        }
    }

    DEBUG_STATUS("DONE");
}
