// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/numpy/functions.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt::tt_metal::LegacyShape shape = {1, 32, 61, 32};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = ttnn::numpy::arange<bfloat16>(0, tt_metal::compute_volume(shape), 1).reshape(shape).to(device);
        Tensor b = ttnn::tilize_with_zero_padding(a);
        Tensor c =  b.cpu();
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        log_debug(LogTest, "Moving src data to host to validate");
        Tensor host_a = a.cpu(); // Move tensor a to host to validate
        Tensor g = Tensor(host_a.get_storage(), shape, DataType::BFLOAT16, Layout::ROW_MAJOR);
        // TODO: Update when tensor.pad_to_tile() function is added
        auto padded_shape = g.get_legacy_shape();
        padded_shape[2] = round_up(padded_shape[2], TILE_HEIGHT);
        padded_shape[3] = round_up(padded_shape[3], TILE_WIDTH);
        Tensor padded_g = g.pad(padded_shape, ttnn::SimpleShape{0,0,0,0}, 0);
        Tensor golden = padded_g.to(Layout::TILE);
        auto golden_vec =  owned_buffer::get_as<bfloat16>(golden);
        auto result_vec = owned_buffer::get_as<bfloat16>(c);
        log_debug(LogTest, "Validating - golden vec size: {} , result vec size {}", golden_vec.size(), result_vec.size());
        uint32_t num_errors = 0;
        for(uint32_t i = 0; i < result_vec.size() ; i++) {
            if(result_vec[i] != golden_vec[i]) {
                if(num_errors < 10)
                    log_error(LogTest, "Error at i={} result={} golden={}", i, result_vec[i].to_float(), golden_vec[i].to_float());
                num_errors++;
            }
        }
        pass &= (result_vec == golden_vec);

	pass &= CloseDevice(device);
    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
