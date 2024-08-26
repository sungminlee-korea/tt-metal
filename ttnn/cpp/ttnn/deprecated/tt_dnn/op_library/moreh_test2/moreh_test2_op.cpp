// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/moreh_test2_op.hpp"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/cpp/ttnn/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {
void MorehTest2::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &input2 = input_tensors.at(1);
    auto &output = output_tensors.at(0);

    check_tensor(input, "moreh_test2", "input");
    check_tensor(input2, "moreh_test2", "input2", {DataType::UINT8});
    check_tensor(output, "moreh_test2", "output");
}

std::vector<Shape> MorehTest2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehTest2::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        log_debug(LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {output_tensors.at(0).value()};
    }

    log_debug(LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehTest2::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input{input_tensors.at(0)};
    const auto &input2{input_tensors.at(1)};
    const auto &output{output_tensors.at(0)};
    const auto &p = this->p;
    const auto &compute_kernel_config = this->compute_kernel_config;

    return moreh_test2_impl(input, input2, output, p, compute_kernel_config);
}

Tensor moreh_test2(
    const Tensor &input,
    const Tensor &input2,
    const std::optional<const Tensor> output,
    const MemoryConfig &output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    float p = 0.5;
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input, input2}))};
    auto kernel_config_val =
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);
    operation::launch_op(
        [p, output_mem_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehTest2{
                    .p = p, .output_mem_config = output_mem_config, .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input, input2},
        output_tensors,
        {},
        {output});
    return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
