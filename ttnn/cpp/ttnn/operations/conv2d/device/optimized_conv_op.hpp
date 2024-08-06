// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/experimental/tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
enum class OptimizedConvOpParallelizationStrategy {
    MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE
};

struct OptimizedConvParallelizationConfig {
    CoreCoord grid_size; // (x,y)
    uint32_t num_cores_nhw;
    uint32_t per_core_out_matrix_height_ntiles;
    uint32_t per_core_out_matrix_width_ntiles;
    // std::size_t in0_block_w;
    // std::size_t out_subblock_h;
    // std::size_t out_subblock_w;
    // std::size_t per_core_M;
    // std::size_t per_core_N;

    CoreCoord get_grid_size() const {
        return this->grid_size;
    }
};


struct OptimizedConvParallelizationConfigNew {
    CoreCoord grid_size; // (x,y)
    uint32_t num_cores_nhw;
    uint32_t per_core_out_matrix_height;
    uint32_t per_core_out_matrix_width;
    // std::size_t in0_block_w;
    // std::size_t out_subblock_h;
    // std::size_t out_subblock_w;
    // std::size_t per_core_M;
    // std::size_t per_core_N;

    CoreCoord get_grid_size() const {
        return this->grid_size;
    }
};

struct OptimizedConvBlockConfig {
    uint32_t act_block_h_ntiles;
    uint32_t act_block_w_ntiles;
    uint32_t out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles;
};

operation::ProgramWithCallbacks multi_core_optimized_conv_(const Tensor& a, const Tensor &b, const Shape& ashape, std::optional<const Tensor> bias, vector<int> conv_params, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, const MathFidelity math_fidelity, const OptimizedConvParallelizationConfig& parallelization_config, const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment, Tensor &output);
operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_(const Tensor& a, const Tensor &b, const Shape& ashape, std::optional<const Tensor> bias, vector<int> conv_params, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, const MathFidelity math_fidelity, const OptimizedConvParallelizationConfig& parallelization_config, const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment, Tensor &output);
operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_(const Tensor& a, const Tensor &b, const Shape& ashape, std::optional<const Tensor> bias, const std::optional<const Tensor> conv_reader_indices, vector<int> conv_params, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, const OptimizedConvParallelizationConfig& parallelization_config, const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment, bool use_shallow_conv_variant, bool transpose_mcast, Tensor &output, DeviceComputeKernelConfig compute_kernel_config, bool enable_act_double_buffer, bool enable_split_reader, bool enable_subblock_padding);
operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_new(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias,
    vector<int> conv_params,
    uint32_t output_channels,
    bool untilize_out, bool fuse_relu, MathFidelity math_fidelity,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    bool use_shallow_conv_variant,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    Tensor& output,
    bool enable_act_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding);

struct OptimizedConv {
    OptimizedConvParallelizationConfig parallelization_config;
    OptimizedConvBlockConfig block_config;

    const std::vector<int> conv_params;
    const uint32_t output_channels;
    bool untilize_out, has_bias, fuse_relu;
    MathFidelity math_fidelity;
    uint32_t extra_padding_for_32B_alignment;
    const MemoryConfig memory_config;
    const DataType dtype;
    Shape input_tensor_shape; // For sharded input, input tensor shape is nonsense
    bool use_shallow_conv_variant;
    bool transpose_mcast;   // default for GS = true, WH = false
    const DeviceComputeKernelConfig compute_kernel_config;
    bool enable_act_double_buffer;
    bool enable_split_reader;
    bool enable_subblock_padding;
    OptimizedConv(const std::vector<int>&c_params,
        uint32_t output_channels, bool untile_out,
        bool has_bias, bool fuse_relu,
        MathFidelity mfidelity, const OptimizedConvParallelizationConfig& p_config,
        const OptimizedConvBlockConfig& b_config,
        uint32_t e_padding_for_32B_alignment,
        MemoryConfig memory_config, DataType dtype, Shape input_tensor_shape, bool use_shallow_conv_variant, bool transpose_mcast, const DeviceComputeKernelConfig compute_kernel_config, bool enable_act_double_buffer, bool enable_split_reader, bool enable_subblock_padding) :
            output_channels(output_channels),
            conv_params(c_params),
            untilize_out(untile_out),
            has_bias(has_bias),
            fuse_relu(fuse_relu),
            math_fidelity(mfidelity),
            parallelization_config(p_config),
            block_config(b_config),
            extra_padding_for_32B_alignment(e_padding_for_32B_alignment),
            memory_config(memory_config), dtype(dtype), input_tensor_shape(input_tensor_shape),
            use_shallow_conv_variant(use_shallow_conv_variant),
            transpose_mcast(transpose_mcast),
            compute_kernel_config(compute_kernel_config),
            enable_act_double_buffer(enable_act_double_buffer),
            enable_split_reader(enable_split_reader),
            enable_subblock_padding(enable_subblock_padding) {}

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;

    operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "parallelization_config",
        "block_config",
        "conv_params",
        "output_channels",
        "untilize_out",
        "has_bias",
        "fuse_relu",
        "math_fidelity",
        "extra_padding_for_32B_alignment",
        "memory_config",
        "dtype",
        "input_tensor_shape",
        "use_shallow_conv_variant",
        "enable_act_double_buffer",
        "enable_split_reader",
        "enable_subblock_padding");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->parallelization_config),
            std::cref(this->block_config),
            std::cref(this->conv_params),
            std::cref(this->output_channels),
            std::cref(this->untilize_out),
            std::cref(this->has_bias),
            std::cref(this->fuse_relu),
            std::cref(this->math_fidelity),
            std::cref(this->extra_padding_for_32B_alignment),
            std::cref(this->memory_config),
            std::cref(this->dtype),
            std::cref(this->input_tensor_shape),
            std::cref(this->use_shallow_conv_variant),
            std::cref(this->enable_act_double_buffer),
            std::cref(this->enable_split_reader),
            std::cref(this->enable_subblock_padding));
    }
};

Tensor optimized_conv(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const std::optional<const Tensor> conv_reader_indices,
    const vector<int> conv_params, uint32_t output_channels,
    bool untilize_out, bool has_bias, bool fuse_relu, MathFidelity math_fidelity,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment=0,
    std::optional<MemoryConfig> memory_config = std::nullopt,
    std::optional<DataType> dtype=std::nullopt,
    std::optional<std::array<std::uint32_t, 4>> input_tensor_shape = std::nullopt,
    bool use_shallow_conv_variant = false,
    bool tranpose_mcast = true,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool enable_act_double_buffer = false,
    bool enable_split_reader = false,
    bool enable_subblock_padding = false
);

// new micro op
struct OptimizedConvNew {
    OptimizedConvParallelizationConfig parallelization_config;
    OptimizedConvBlockConfig block_config;
    const std::vector<int> conv_params;
    const uint32_t output_channels;
    bool untilize_out, has_bias, fuse_relu;
    MathFidelity math_fidelity;
    uint32_t extra_padding_for_32B_alignment;
    MemoryConfig memory_config;
    const DataType dtype;
    std::array<std::uint32_t, 4> input_tensor_shape; // For sharded input, input tensor shape is nonsense
    bool use_shallow_conv_variant;
    const DeviceComputeKernelConfig compute_kernel_config;
    bool enable_act_double_buffer;
    bool enable_split_reader;
    bool enable_subblock_padding;
    OptimizedConvNew(const vector<int>& c_params,
        uint32_t output_channels, bool untile_out,
        bool has_bias, bool fuse_relu,
        MathFidelity mfidelity, const OptimizedConvParallelizationConfig& p_config,
        const OptimizedConvBlockConfig& b_config,
        uint32_t e_padding_for_32B_alignment, MemoryConfig out_mem_config,
        DataType dtype,
        std::array<std::uint32_t, 4> input_tensor_shape, bool use_shallow_conv_variant,
        const DeviceComputeKernelConfig compute_kernel_config, bool enable_act_double_buffer, bool enable_split_reader, bool enable_subblock_padding) :
            output_channels(output_channels),
            conv_params(c_params),
            untilize_out(untile_out),
            has_bias(has_bias),
            fuse_relu(fuse_relu),
            math_fidelity(mfidelity),
            parallelization_config(p_config),
            block_config(b_config),
            extra_padding_for_32B_alignment(e_padding_for_32B_alignment),
            memory_config(out_mem_config),
            dtype(dtype), input_tensor_shape(input_tensor_shape),
            use_shallow_conv_variant(use_shallow_conv_variant),
            compute_kernel_config(compute_kernel_config),
            enable_act_double_buffer(enable_act_double_buffer),
            enable_split_reader(enable_split_reader),
            enable_subblock_padding(enable_subblock_padding) {}

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;

    operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "parallelization_config",
        "block_config",
        "conv_params",
        "output_channels",
        "untilize_out",
        "has_bias",
        "fuse_relu",
        "math_fidelity",
        "extra_padding_for_32B_alignment",
        "dtype",
        "input_tensor_shape",
        "use_shallow_conv_variant",
        "enable_act_double_buffer",
        "enable_split_reader",
        "enable_subblock_padding");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->parallelization_config),
            std::cref(this->block_config),
            std::cref(this->conv_params),
            std::cref(this->output_channels),
            std::cref(this->untilize_out),
            std::cref(this->has_bias),
            std::cref(this->fuse_relu),
            std::cref(this->math_fidelity),
            std::cref(this->extra_padding_for_32B_alignment),
            std::cref(this->dtype),
            std::cref(this->input_tensor_shape),
            std::cref(this->use_shallow_conv_variant),
            std::cref(this->enable_act_double_buffer),
            std::cref(this->enable_split_reader),
            std::cref(this->enable_subblock_padding));
    }
};

Tensor optimized_conv_new(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias,
    const vector<int> conv_params,
    uint32_t output_channels,
    bool untilize_out, bool fuse_relu, MathFidelity math_fidelity,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment,
    MemoryConfig memory_config,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    bool use_shallow_conv_variant,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool enable_act_double_buffer = false,
    bool enable_split_reader = false,
    bool enable_subblock_padding = false
);

}  // namespace tt_metal

}  // namespace tt


namespace optimized_conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

pair<uint32_t, uint32_t> compute_opt_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t padding_for_32B_alignment=0);

pair<vector<uint32_t>, vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(Shape conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t padding_for_32B_alignment);

} // optimized_conv_op_utils
