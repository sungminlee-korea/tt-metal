// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"


void kernel_main() {
    uint32_t i = 0;
    uint32_t param_in_addr = get_arg_val<uint32_t>(i); i++;
    uint32_t grad_addr = get_arg_val<uint32_t>(i); i++;
    uint32_t momentum_in_addr = get_arg_val<uint32_t>(i); i++;
    uint32_t num_tiles = get_arg_val<uint32_t>(i); i++;
    uint32_t tile_offset = get_arg_val<uint32_t>(i); i++;
    uint32_t lr = get_arg_val<uint32_t>(i); i++;
    uint32_t momentum = get_arg_val<uint32_t>(i); i++;
    uint32_t dampening = get_arg_val<uint32_t>(i); i++;
    uint32_t weight_decay = get_arg_val<uint32_t>(i); i++;
    uint32_t one = get_arg_val<uint32_t>(i); i++;

    constexpr auto cb_param_in = tt::CB::cb_0;
    constexpr auto cb_grad = tt::CB::cb_1;
    constexpr auto cb_momentum_in = tt::CB::cb_2;

    constexpr auto cb_scalar_args = tt::CB::cb_24;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    auto param_in = InterleavedAddrGenFastHelper(param_in_addr, cb_param_in, 0);
    auto grad = InterleavedAddrGenFastHelper(grad_addr, cb_grad, 1);

    #if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    auto momentum_in = InterleavedAddrGenFastHelper(momentum_in_addr, cb_momentum_in, 2);
    #endif

    fill_cb_with_value(cb_scalar_args, lr);
    fill_cb_with_value(cb_scalar_args, momentum);
    fill_cb_with_value(cb_scalar_args, dampening);
    fill_cb_with_value(cb_scalar_args, weight_decay);
    fill_cb_with_value(cb_scalar_args, one);

    uint32_t l1_write_addr;

    uint32_t curr_tile = tile_offset;

    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        // param_in
        noc_async_read_tile_helper(cb_param_in, onetile, curr_tile, param_in);

        // grad
        noc_async_read_tile_helper(cb_grad, onetile, curr_tile, grad);

        // momentum
        #if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
            noc_async_read_tile_helper(cb_momentum_in, onetile, curr_tile, momentum_in);
        #endif
        curr_tile++;
    }
}
