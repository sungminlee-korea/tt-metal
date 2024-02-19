# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def report_results(test_case_name, total_data_transferred):
    print("reporting results...")
    setup = device_post_proc_config.default_setup()
    setup.timerAnalysis = {
        "LATENCY": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "timerID": 80},
            "end": {"risc": "ERISC", "timerID": 3},
        },
    }

    throughput_GB_per_second = -1
    os.system("sed -i '/^[[:space:]]*$/d' ./generated/profiler/.logs/profile_log_device.csv")
    try:
        setup.deviceInputLog = "./generated/profiler/.logs/profile_log_device.csv"
        stats = import_log_run_stats(setup)
        core = [key for key in stats["devices"][1]["cores"].keys() if key != "DEVICE"][0]
        # test_cycles = stats["devices"][1]["cores"][core]["riscs"]["ERISC"]["analysis"]["LATENCY"]["stats"]["First"]
        test_cycles = stats["devices"][1]["cores"][core]["riscs"]["TENSIX"]["analysis"]["LATENCY"]["stats"]["First"]
        total_bytes = total_data_transferred
        cycles_per_second = 1000000000
        throughput_bytes_per_second = total_bytes / (test_cycles / 1000000000)
        throughput_GB_per_second = throughput_bytes_per_second / (1000 * 1000 * 1000)
        print(
            f"Cycles: {test_cycles}, Total Data Sent(B): {total_data_transferred}, Throughput: {throughput_GB_per_second} GB/s"
        )
    except:
        print("Error in results parsing")
        assert False

    os.system(f"rm -rf ./generated/profiler/.logs/{test_case_name}/")
    os.system(f"mkdir -p ./generated/profiler/.logs/{test_case_name}/")
    os.system(
        f"cp ./generated/profiler/.logs/profile_log_device.csv ./generated/profiler/.logs/{test_case_name}/profile_log_device.csv"
    )

    return throughput_GB_per_second


@pytest.mark.parametrize("num_messages_to_send", [1, 2, 4, 8, 16, 326, 680, 1000])
@pytest.mark.parametrize(
    "buffer_size_bytes",
    [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16 * 1024,
        24 * 1024,
        32 * 1024,
        50 * 1024,
        64 * 1024,
        128 * 1024,
    ],
)
@pytest.mark.parametrize("num_sends_before_sync", [1, 2, 4, 8])
def test_ethernet_send_data_microbenchmark(num_messages_to_send, buffer_size_bytes, num_sends_before_sync):
    print(
        f"test_ethernet_send_data_microbenchmark - buffer_size_bytes: {buffer_size_bytes}, num_messages_to_send: {num_messages_to_send}, num_sends_before_sync: {num_sends_before_sync}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_sends_before_sync}  \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping.cpp\" > /dev/null 2>&1"
    )
    if rc != 0:
        print("Error in running the test")
        assert False
    return report_results(
        f"test_ethernet_send_data_microbenchmark_{num_messages_to_send}_{buffer_size_bytes}_{num_sends_before_sync}",
        num_messages_to_send * buffer_size_bytes,
    )


@pytest.mark.parametrize("num_messages_to_send", [1, 2, 4, 8, 16, 100, 326, 680, 1000])
@pytest.mark.parametrize(
    "buffer_size_bytes",
    [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16 * 1024, 24 * 1024, 32 * 1024, 50 * 1024, 64 * 1024, 128 * 1024],
)
@pytest.mark.parametrize("num_transaction_buffers", [1, 2, 4, 8])
def test_ethernet_send_data_microbenchmark_concurrent(num_messages_to_send, buffer_size_bytes, num_transaction_buffers):
    print(
        f"test_ethernet_send_data_microbenchmark - buffer_size_bytes: {buffer_size_bytes}, num_messages_to_send: {num_messages_to_send}, num_transaction_buffers: {num_transaction_buffers}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        # f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_transaction_buffers} \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping_multi_channel.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\" > /dev/null 2>&1"
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_transaction_buffers} \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping_multi_channel.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\""
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    return report_results(
        f"test_ethernet_send_data_microbenchmark_concurrent_{num_messages_to_send}_{buffer_size_bytes}_{num_transaction_buffers}",
        num_messages_to_send * buffer_size_bytes,
    )


@pytest.mark.parametrize(
    "input_buffer_size_bytes",
    [1024, 2048, 4096, 16384, 64 * 1024, 128 * 1024, 256 * 1024, 835584, 16711680, 680 * 32768],
)
@pytest.mark.parametrize(
    "eth_buffer_size_bytes",
    [1024, 2048, 4096, 8192, 12 * 1024, 16 * 1024, 14 * 1024, 20 * 1024, 32 * 1024, 50 * 1024],
)
@pytest.mark.parametrize("num_transaction_buffers", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("input_buffer_page_size", [1024, 2048, 4096, 8192])
# @pytest.mark.parametrize("precomputed_address_buffer_size", [0, 16, 32])
def test_ethernet_send_data_microbenchmark_concurrent_with_dram_read(
    input_buffer_size_bytes,
    eth_buffer_size_bytes,
    num_transaction_buffers,
    input_buffer_page_size  # ,
    # precomputed_address_buffer_size,
):
    if eth_buffer_size_bytes < input_buffer_page_size:
        pytest.skip("eth_buffer_size_bytes < input_buffer_page_size")
        return

    if input_buffer_size_bytes < input_buffer_page_size:
        pytest.skip("input_buffer_size_bytes < input_buffer_page_size")
        return

    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            input_buffer_page_size: {input_buffer_page_size}, \
                num_transaction_buffers: {num_transaction_buffers}, \
                    eth_buffer_size_bytes: {eth_buffer_size_bytes} \
                        input_buffer_size_bytes: {input_buffer_size_bytes}"
    print(
        f"test_ethernet_send_data_microbenchmark - \
            input_buffer_page_size: {input_buffer_page_size}, \
                num_transaction_buffers: {num_transaction_buffers}, \
                    eth_buffer_size_bytes: {eth_buffer_size_bytes} \
                        input_buffer_size_bytes: {input_buffer_size_bytes}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_read_and_send_data \
            \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_forward_local_chip_data_looping_multi_channel.cpp\" \
            \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\" \
            {eth_buffer_size_bytes} \
            {num_transaction_buffers} \
            {input_buffer_page_size} \
            {input_buffer_size_bytes} \
            {1} \
            {1} \
            0 \
            "
        # > /dev/null 2>&1"
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    test_string_name = f"test_ethernet_send_data_microbenchmark_concurrent_with_dram_read_"
    test_string_name += f"{input_buffer_page_size}_"
    test_string_name += f"{num_transaction_buffers}_"
    test_string_name += f"{eth_buffer_size_bytes}_"
    test_string_name += f"{input_buffer_size_bytes}"  # _"
    # test_string_name += f"{precomputed_address_buffer_size}"
    return report_results(test_string_name, input_buffer_size_bytes)


@pytest.mark.parametrize(
    "input_buffer_size_bytes",
    [1024, 2048, 4096, 16384, 64 * 1024, 128 * 1024, 256 * 1024, 835584, 16711680, 680 * 32768],
)
@pytest.mark.parametrize(
    "eth_buffer_size_bytes",
    [1024, 2048, 4096, 8192, 12 * 1024, 16 * 1024, 14 * 1024, 20 * 1024, 32 * 1024, 50 * 1024],
)
@pytest.mark.parametrize("num_transaction_buffers", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("input_buffer_page_size", [1024, 2048, 4096, 8192])
# @pytest.mark.parametrize("precomputed_address_buffer_size", [0, 16, 32])
def test_ethernet_send_data_microbenchmark_concurrent_with_dram_read_and_write(
    input_buffer_size_bytes,
    eth_buffer_size_bytes,
    num_transaction_buffers,
    input_buffer_page_size  # ,
    # precomputed_address_buffer_size,
):
    if eth_buffer_size_bytes < input_buffer_page_size:
        pytest.skip("eth_buffer_size_bytes < input_buffer_page_size")
        return

    if input_buffer_size_bytes < input_buffer_page_size:
        pytest.skip("input_buffer_size_bytes < input_buffer_page_size")
        return

    test_string_name = f"test_ethernet_send_data_microbenchmark - \
            input_buffer_page_size: {input_buffer_page_size}, \
                num_transaction_buffers: {num_transaction_buffers}, \
                    eth_buffer_size_bytes: {eth_buffer_size_bytes} \
                        input_buffer_size_bytes: {input_buffer_size_bytes}"
    print(
        f"test_ethernet_send_data_microbenchmark - \
            input_buffer_page_size: {input_buffer_page_size}, \
                num_transaction_buffers: {num_transaction_buffers}, \
                    eth_buffer_size_bytes: {eth_buffer_size_bytes} \
                        input_buffer_size_bytes: {input_buffer_size_bytes}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_read_and_send_data \
            \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_forward_local_chip_data_looping_multi_channel.cpp\" \
            \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_non_blocking_receive_fwd_to_dram.cpp\" \
            {eth_buffer_size_bytes} \
            {num_transaction_buffers} \
            {input_buffer_page_size} \
            {input_buffer_size_bytes} \
            {1} \
            {1} \
            0 \
            "
        # > /dev/null 2>&1"
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    test_string_name = f"test_ethernet_send_data_microbenchmark_concurrent_with_dram_read_"
    test_string_name += f"{input_buffer_page_size}_"
    test_string_name += f"{num_transaction_buffers}_"
    test_string_name += f"{eth_buffer_size_bytes}_"
    test_string_name += f"{input_buffer_size_bytes}"  # _"
    # test_string_name += f"{precomputed_address_buffer_size}"
    return report_results(test_string_name, input_buffer_size_bytes)


def test_run_ethernet_send_data_microbenchmark_sweep():
    input_buffer_size_bytes = [16384, 64 * 1024, 256 * 1024, 16711680]  # , 680*32768]
    eth_buffer_size_bytes = [8192, 16 * 1024, 32 * 1024, 50 * 1024]
    # num_transaction_buffers = [1, 2, 3, 4, 8]
    # num_transaction_buffers = [2, 3, 4, 8]
    num_transaction_buffers = [3, 4, 8]
    input_buffer_page_size = [1024, 2048, 4096]

    recorded_throughput_slow_mode = {}
    recorded_throughput_concurrent = {}
    recorded_throughputs = {}

    for page_size in input_buffer_page_size:
        for input_size_bytes in input_buffer_size_bytes:
            for eth_l1_buffer_size in eth_buffer_size_bytes:
                for max_concurrent_transfers in num_transaction_buffers:
                    if eth_buffer_size_bytes < input_buffer_page_size:
                        continue

                    if input_buffer_size_bytes < input_buffer_page_size:
                        continue

                    if eth_l1_buffer_size * max_concurrent_transfers > 160000:
                        continue

                    attempts = 0
                    max_attempts = 10
                    successful = False
                    while not successful and attempts < max_attempts:
                        try:
                            throughput = test_ethernet_send_data_microbenchmark_concurrent_with_dram_read_and_write(
                                input_size_bytes, eth_l1_buffer_size, max_concurrent_transfers, page_size
                            )
                            successful = True
                        except:
                            attempts += 1
                            continue

                    if page_size not in recorded_throughputs:
                        recorded_throughputs[page_size] = {}

                    if input_size_bytes not in recorded_throughputs[page_size]:
                        recorded_throughputs[page_size][input_size_bytes] = {}

                    if eth_l1_buffer_size not in recorded_throughputs[page_size][input_size_bytes]:
                        recorded_throughputs[page_size][input_size_bytes][eth_l1_buffer_size] = {}

                    recorded_throughputs[page_size][input_size_bytes][eth_l1_buffer_size][
                        max_concurrent_transfers
                    ] = throughput

    ## Print out the results

    COLUMN_WIDTH = 10
    for page_size in recorded_throughputs:
        print(f"##############################################")
        print(f"##############################################")
        print(f"PAGE_SIZE: {page_size}")
        for input_size_bytes in recorded_throughputs[page_size]:
            print(f"||----------------------------------------------")
            print(f"|| INPUT_SIZE: {input_size_bytes}")
            print(f"||\t NUM_TRANSACTIONS:")

            space = ""
            top_row = f"||{space:^{COLUMN_WIDTH}}"
            for num_concurrent_transfers in num_transaction_buffers:
                top_row += f"|{num_concurrent_transfers:^{COLUMN_WIDTH}}"
            print(top_row)

            for eth_l1_buffer_size in recorded_throughputs[page_size][input_size_bytes]:
                row = f"||{eth_l1_buffer_size:<{COLUMN_WIDTH}}"
                for num_concurrent_transfers in num_transaction_buffers:
                    if (
                        num_concurrent_transfers
                        not in recorded_throughputs[page_size][input_size_bytes][eth_l1_buffer_size]
                    ):
                        row += f"|{'N/A':<{COLUMN_WIDTH}}"
                        continue
                    else:
                        throughput = recorded_throughputs[page_size][input_size_bytes][eth_l1_buffer_size][
                            num_concurrent_transfers
                        ]
                        row += f"|{throughput:<{COLUMN_WIDTH}.2f}"

                print(row)

    print(f"##############################################")
    print(f"page_size,input_size_bytes,eth_l1_buffer_size,num_concurrent_transfers,throughput(GBps)")
    for page_size, throughputs_input_size_bytes in recorded_throughputs.items():
        for input_size_bytes, throughputs_eth_1l_buf_size_bytes in throughputs_input_size_bytes.items():
            for eth_l1_buffer_size, throughputs_max_concurrent_transfers in throughputs_eth_1l_buf_size_bytes.items():
                for num_concurrent_transfers, throughput in throughputs_max_concurrent_transfers.items():
                    print(
                        f"{page_size},{input_size_bytes},{eth_l1_buffer_size},{num_concurrent_transfers},{throughput}"
                    )
