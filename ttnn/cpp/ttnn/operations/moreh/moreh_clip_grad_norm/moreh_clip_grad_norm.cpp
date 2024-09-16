// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm.hpp"

#include <cmath>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "device/moreh_clip_grad_norm_step1_device_operation.hpp"
#include "device/moreh_clip_grad_norm_step2_device_operation.hpp"
#include "device/moreh_clip_grad_norm_step3_device_operation.hpp"
#include "ttnn/cpp/ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

namespace {
inline uint32_t get_num_device_cores(Device* device) {
    const auto num_cores_x = static_cast<uint32_t>(device->compute_with_storage_grid_size().x);
    const auto num_cores_y = static_cast<uint32_t>(device->compute_with_storage_grid_size().y);
    return num_cores_x * num_cores_y;
}
}  // namespace

void moreh_clip_grad_norm_step1(const std::vector<Tensor>& inputs, float norm_type, const Tensor& tmp_pow_sum) {
    const auto max_num_inputs = get_num_device_cores(inputs.at(0).device());
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());
    const auto num_iter = (total_num_inputs + max_num_inputs - 1) / max_num_inputs;

    uint32_t tile_offset{0};
    auto num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; ++i) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);
        ttnn::prim::moreh_clip_grad_norm_step1(
            std::vector<Tensor>(inputs.begin() + tile_offset, inputs.begin() + tile_offset + num_inputs_at_this_iter),
            norm_type,
            tile_offset,
            tmp_pow_sum);
        if (i < (num_iter - 1)) {
            tile_offset += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }
}

void moreh_clip_grad_norm_step2(const Tensor& tmp_pow_sum, float norm_type, const Tensor& total_norm) {
    ttnn::prim::moreh_clip_grad_norm_step2(tmp_pow_sum, norm_type, total_norm);
}

void moreh_clip_grad_norm_step3(const std::vector<Tensor>& inputs, const Tensor& clip_coef_clamped) {
    const auto max_num_inputs = get_num_device_cores(inputs.at(0).device());
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());
    const auto num_iter = (total_num_inputs + max_num_inputs - 1) / max_num_inputs;

    uint32_t start_input_idx{0};
    auto num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; ++i) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);

        ttnn::prim::moreh_clip_grad_norm_step3(
            std::vector<Tensor>(
                inputs.begin() + start_input_idx, inputs.begin() + start_input_idx + num_inputs_at_this_iter),
            clip_coef_clamped);

        if (i < (num_iter - 1)) {
            start_input_idx += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }
}

Tensor moreh_clip_grad_norm_impl(
    const std::vector<Tensor>& inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const Tensor& tmp_pow_sum,
    const Tensor& total_norm) {
    // Sum[|e|^p]
    moreh_clip_grad_norm_step1(inputs, norm_type, tmp_pow_sum);

    // Sum[Sum[|e|^p]]^(1/p)
    ttnn::prim::moreh_clip_grad_norm_step2(tmp_pow_sum, norm_type, total_norm);

    if (error_if_nonfinite) {
        const auto fp32_total_norm =
            tensor_impl::cast_vec<float>(owned_buffer::get_as<class bfloat16>(total_norm.cpu())).at(0);
        TT_FATAL(
            std::isfinite(fp32_total_norm),
            "The total norm of order {} for gradients from `parameters` is non-finite, so it cannot be "
            "clipped. To disable this error and scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`",
            norm_type);
    }

    // max_norm / (total_norm + 1e-6)
    const auto& clip_coef = ttnn::multiply(ttnn::add(total_norm, 1e-6f), (1 / max_norm));

    // min(clip_coef, 1.0f)
    Tensor scalar =
        ttnn::operations::creation::create_scalar(1.0f, inputs.at(0).get_dtype(), Layout::TILE, inputs.at(0).device());
    const auto& clip_coef_clamped = ttnn::minimum(clip_coef, scalar);
    scalar.deallocate();

    // Inplace update inputs(inputs *= clip_coef_clamped)
    ttnn::prim::moreh_clip_grad_norm_step3(inputs, clip_coef_clamped);

    return total_norm;
}

Tensor MorehClipGradNorm::invoke(
    const std::vector<Tensor>& inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const std::optional<Tensor>& total_norm,
    const MemoryConfig& memory_config) {
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());
    tt::tt_metal::Shape tmp_pow_sum_shape{
        1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH * total_num_inputs};
    const auto& tmp_pow_sum =
        create_device_tensor(tmp_pow_sum_shape, inputs.at(0).get_dtype(), Layout::TILE, inputs.at(0).device());
    if (total_norm.has_value() && (total_norm != std::nullopt)) {
        return moreh_clip_grad_norm_impl(
            inputs, max_norm, norm_type, error_if_nonfinite, tmp_pow_sum, total_norm.value());
    }

    Padding padding{
        {{0, 0}, {0, 0}, {0, tt::constants::TILE_HEIGHT - 1}, {0, tt::constants::TILE_WIDTH - 1}},
        Padding::PadValue::Zero};
    tt::tt_metal::Shape total_norm_shape{{1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}, padding};
    const auto& created_total_norm = create_device_tensor(
        total_norm_shape, inputs.at(0).get_dtype(), Layout::TILE, inputs.at(0).device(), memory_config);
    return moreh_clip_grad_norm_impl(inputs, max_norm, norm_type, error_if_nonfinite, tmp_pow_sum, created_total_norm);
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
