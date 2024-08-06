// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace conv2d {

void py_module(py::module& module) {

    module.def("conv", &conv, R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");
    module.def(
        "conv2d",
        [](const ttnn::Tensor& input_tensor,
            const ttnn::Tensor& weight_tensor,
            ttnn::Device& device,
            uint32_t in_channels,
            uint32_t out_channels,
            uint32_t batch_size,
            uint32_t input_height,
            uint32_t input_width,
            std::array<uint32_t, 2> kernel_size,
            std::array<uint32_t, 2> stride,
            std::array<uint32_t, 2> padding,
            std::array<uint32_t, 2> dilation,
            uint32_t groups,
            std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
            std::optional<const Conv2dConfig> conv_config_ = std::nullopt) -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
            return ttnn::operations::conv2d::conv2d(
                input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation,
                    groups, bias_tensor, conv_config_);
        },
        py::kw_only(),
        py::arg("input_tensor"),
        py::arg("weight_tensor"),
        py::arg("device"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"),
        py::arg("bias_tensor") = std::nullopt,
        py::arg("conv_config") = std::nullopt);

    module.def(
        "optimized_conv",
        &optimized_conv,
        py::arg("a").noconvert(),
        py::arg("b").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("conv_reader_indices").noconvert() = std::nullopt,
        py::arg("conv_params").noconvert(),
        py::arg("output_channels").noconvert(),
        py::arg("untilize_out").noconvert(),
        py::arg("has_bias").noconvert(),
        py::arg("fuse_relu").noconvert(),
        py::arg("math_fidelity").noconvert(),
        py::arg("parallelization_config").noconvert(),
        py::arg("block_config").noconvert(),
        py::arg("extra_padding_for_32_B_alignment").noconvert() = 0,
        py::arg("memory_config").noconvert() = std::nullopt,
        py::arg("dtype").noconvert() = std::nullopt,
        py::arg("input_tensor_shape").noconvert() = std::nullopt,
        py::arg("use_shallow_conv_variant").noconvert() = false,
        py::arg("transpose_mcast").noconvert() = true,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("enable_act_double_buffer").noconvert() = false,
        py::arg("enable_split_reader").noconvert() = false,
        py::arg("enable_subblock_padding").noconvert() = false,
        R"doc(
        Perform a conv ``A x B`` with two tensors
        This op tilizes tensor A and untilizes the output

        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | Argument     | Description                                                                                | Data type | Valid range | Required |
        +==============+============================================================================================+===========+=============+==========+
        | a            | Conv activation TT tensor (CHANNELS LAST                                                   | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | b            | Conv weight TT tensor (TILED)                                                              | Tensor    |             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
        | conv_params  | Conv parameters list: kernel size H, kernel size W ,stride H,stride W,pad H,pad W          |Vector<int>|             | Yes      |
        +--------------+--------------------------------------------------------------------------------------------+-----------+-------------+----------+
    )doc");

    module.def(
        "get_conv_padded_input_shape_and_mem_config",
        [](ttnn::Device& device,
            const ttnn::Tensor& input_tensor,
            const Conv2dConfig& conv_config,
            uint32_t batch_size,
            uint32_t height,
            uint32_t width,
            uint32_t in_channels,
            uint32_t out_channels) -> std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> {
            return ttnn::operations::conv2d::get_conv_padded_input_shape_and_mem_config(
                device, input_tensor, conv_config, batch_size, height, width, in_channels, out_channels);
        },
        py::kw_only(),
        py::arg("device"),
        py::arg("input_tensor"),
        py::arg("conv_config"),
        py::arg("batch_size"),
        py::arg("height"),
        py::arg("width"),
        py::arg("in_channels"),
        py::arg("out_channels"));
    auto py_conv_config = py::class_<Conv2dConfig>(module, "Conv2dConfig");
    py_conv_config.def(
            py::init<MathFidelity, DataType, DataType, bool, bool, bool, string, uint32_t, bool, bool, uint32_t, bool, bool, bool, std::optional<CoreRangeSet>, bool, Layout, bool, bool, bool>(),
            py::kw_only(),
            py::arg("math_fidelity") = MathFidelity::HiFi4,
            py::arg("dtype") = DataType::BFLOAT16,
            py::arg("weights_dtype") = DataType::BFLOAT16,
            py::arg("math_approx_mode_enabled") = true,
            py::arg("fp32_dest_acc_enabled") = false,
            py::arg("packer_l1_accum_enabled") = false,
            py::arg("activation") = "",
            py::arg("input_channels_alignment") = 32,
            py::arg("deallocate_activation") = false,
            py::arg("reallocate_halo_output") = false,
            py::arg("act_block_h_override") = 0,
            py::arg("reshard_if_not_optimal") = false,
            py::arg("override_sharding_config") = false,
            py::arg("height_sharding") = true,
            py::arg("core_grid") = std::nullopt,
            py::arg("transpose_shards") = true,
            py::arg("output_layout") = Layout::TILE,
            py::arg("enable_act_double_buffer") = false,
            py::arg("enable_split_reader") = false,
            py::arg("enable_subblock_padding") = false
        );
        py_conv_config.def_readwrite("math_fidelity", &Conv2dConfig::math_fidelity);
        py_conv_config.def_readwrite("dtype", &Conv2dConfig::dtype);
        py_conv_config.def_readwrite("weights_dtype", &Conv2dConfig::weights_dtype);
        py_conv_config.def_readwrite("math_approx_mode_enabled", &Conv2dConfig::math_approx_mode_enabled);
        py_conv_config.def_readwrite("fp32_dest_acc_enabled", &Conv2dConfig::fp32_dest_acc_enabled);
        py_conv_config.def_readwrite("packer_l1_accum_enabled", &Conv2dConfig::packer_l1_accum_enabled);
        py_conv_config.def_readwrite("activation", &Conv2dConfig::activation);
        py_conv_config.def_readwrite("input_channels_alignment", &Conv2dConfig::input_channels_alignment);
        py_conv_config.def_readwrite("deallocate_activation", &Conv2dConfig::deallocate_activation);
        py_conv_config.def_readwrite("reallocate_halo_output", &Conv2dConfig::reallocate_halo_output);
        py_conv_config.def_readwrite("act_block_h_override", &Conv2dConfig::act_block_h_override);
        py_conv_config.def_readwrite("reshard_if_not_optimal", &Conv2dConfig::reshard_if_not_optimal);
        py_conv_config.def_readwrite("override_sharding_config", &Conv2dConfig::override_sharding_config);
        py_conv_config.def_readwrite("height_sharding", &Conv2dConfig::height_sharding);
        py_conv_config.def_readwrite("core_grid", &Conv2dConfig::core_grid);
        py_conv_config.def_readwrite("transpose_shards", &Conv2dConfig::transpose_shards);
        py_conv_config.def_readwrite("output_layout", &Conv2dConfig::output_layout);
        py_conv_config.def_readwrite("enable_act_double_buffer", &Conv2dConfig::enable_act_double_buffer);
        py_conv_config.def_readwrite("enable_split_reader", &Conv2dConfig::enable_split_reader);
        py_conv_config.def_readwrite("enable_subblock_padding", &Conv2dConfig::enable_subblock_padding);

    py::class_<OptimizedConvParallelizationConfig>(module, "OptimizedConvParallelizationConfig")
        .def(
            py::init<CoreCoord, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("grid_size"),
            py::arg("num_cores_nhw"),
            py::arg("per_core_out_matrix_height_ntiles").noconvert(),
            py::arg("per_core_out_matrix_width_ntiles").noconvert())
        .def_property_readonly("grid_size", [](OptimizedConvParallelizationConfig const& c) { return c.grid_size; })
        .def_property_readonly(
            "num_cores_nhw", [](OptimizedConvParallelizationConfig const& c) { return c.num_cores_nhw; })
        .def_property_readonly(
            "per_core_out_matrix_height_ntiles",
            [](OptimizedConvParallelizationConfig const& c) { return c.per_core_out_matrix_height_ntiles; })
        .def_property_readonly("per_core_out_matrix_width_ntiles", [](OptimizedConvParallelizationConfig const& c) {
            return c.per_core_out_matrix_width_ntiles;
        });

    py::class_<OptimizedConvParallelizationConfigNew>(module, "OptimizedConvParallelizationConfigNew")
        .def(
            py::init<CoreCoord, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("grid_size"),
            py::arg("num_cores_nhw"),
            py::arg("per_core_out_matrix_height").noconvert(),
            py::arg("per_core_out_matrix_width").noconvert())
        .def_property_readonly("grid_size", [](OptimizedConvParallelizationConfigNew const& c) { return c.grid_size; })
        .def_property_readonly(
            "num_cores_nhw", [](OptimizedConvParallelizationConfigNew const& c) { return c.num_cores_nhw; })
        .def_property_readonly(
            "per_core_out_matrix_height",
            [](OptimizedConvParallelizationConfigNew const& c) { return c.per_core_out_matrix_height; })
        .def_property_readonly("per_core_out_matrix_width", [](OptimizedConvParallelizationConfigNew const& c) {
            return c.per_core_out_matrix_width;
        });

    py::class_<OptimizedConvBlockConfig>(module, "OptimizedConvBlockConfig")
        .def(
            py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
            py::kw_only(),
            py::arg("act_block_h_ntiles").noconvert(),
            py::arg("act_block_w_ntiles").noconvert(),
            py::arg("out_subblock_h_ntiles").noconvert(),
            py::arg("out_subblock_w_ntiles").noconvert())
        .def_property_readonly(
            "act_block_h_ntiles", [](OptimizedConvBlockConfig const& c) { return c.act_block_h_ntiles; })
        .def_property_readonly(
            "act_block_w_ntiles", [](OptimizedConvBlockConfig const& c) { return c.act_block_w_ntiles; })
        .def_property_readonly(
            "out_subblock_h_ntiles", [](OptimizedConvBlockConfig const& c) { return c.out_subblock_h_ntiles; })
        .def_property_readonly(
            "out_subblock_w_ntiles", [](OptimizedConvBlockConfig const& c) { return c.out_subblock_w_ntiles; });

}

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
