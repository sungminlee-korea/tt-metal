// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "index_fill.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/index_fill/device/index_fill_device_operation.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::index_fill {

using IndexFillType = decltype(ttnn::index_fill);

void bind_index_fill_operation(py::module& module) {

    bind_registered_operation(
        module,
        ttnn::index_fill,
        "Index fill Operation",
        ttnn::pybind_overload_t{
            [](const IndexFillType& self,
                const Tensor &input,
                const Tensor &index,
                const uint32_t dim,
                const std::variant<float, int> value,
                const std::optional<Tensor> &output,
                const std::optional<MemoryConfig> &memory_config) -> Tensor {
                return self(input, index, dim, value, output, memory_config);
            },
            py::arg("input"),
            py::arg("index"),
            py::arg("dim"),
            py::arg("value"),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::index_fill
