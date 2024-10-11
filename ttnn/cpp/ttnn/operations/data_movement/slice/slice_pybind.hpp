// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "slice.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_slice(py::module& module) {
    auto doc =
        R"doc(
            slice(input_tensor: ttnn.Tensor, slice_start: List[int[tensor rank], slice_end: List[int[tensor rank],  value: Union[int, float], *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Returns a sliced tensor. If the input tensor is on host, the slice will be performed on host, and if its on device it will be performed on device.

            Equivalent pytorch code:

            .. code-block:: python

                output_tensor = input_tensor[output_start: output_end]

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`slice_start`: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                * :attr:`slice_end`: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
                * :attr:`step` (Optional[List[int[tensor rank]]): Step size for each dim. Default is None, which works out be 1 for each dimension.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    // TODO: implementing the array version and overloading the pybind with all the possible array sizes is better than a vector with a fixed size default value
    using OperationType = decltype(ttnn::slice);
    ttnn::bind_registered_operation(
        module,
        ttnn::slice,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const std::vector<int> &slice_start,
                const std::vector<int> &slice_end,
                const std::optional<std::vector<int>> &step,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                const std::optional<Tensor>& optional_output_tensor,
                uint8_t queue_id) {
                    const auto step_value = step.value_or(std::vector<int>(slice_end.size(), 1));
                    return self(queue_id, input_tensor, slice_start, slice_end, step_value, memory_config, optional_output_tensor);
                },
                py::arg("input_tensor"),
                py::arg("slice_start"),
                py::arg("slice_end"),
                py::arg("slice_step") = std::nullopt, // should consider a better default value
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0,
                },

        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const std::array<uint32_t, 4> &begins,
                const std::array<uint32_t, 4> &ends,
                const std::array<uint32_t, 4> &step,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                const std::optional<Tensor>& optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, begins, ends, step, memory_config, optional_output_tensor);
                },
                py::arg("input_tensor"),
                py::arg("starts"),
                py::arg("ends"),
                py::arg("steps"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0,
                }
        );
}
}  // namespace ttnn::operations::data_movement::detail
