// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sgd_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_sgd/moreh_sgd.hpp"

namespace ttnn::operations::moreh::moreh_sgd {
void bind_moreh_sgd_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_sgd,
        "Moreh Sgd Operation",
        ttnn::pybind_arguments_t{
            py::arg("param_in"),
            py::arg("grad"),
            py::arg("momentum_buffer_in") = std::nullopt,
            py::arg("param_out") = std::nullopt,
            py::arg("momentum_buffer_out") = std::nullopt,

            py::arg("lr"),
            py::arg("momentum"),
            py::arg("dampening"),
            py::arg("weight_decay"),
            py::arg("nesterov"),
            py::arg("momentum_initialized"),

            py::arg("param_out_mem_config") = std::nullopt,
            py::arg("momentum_buffer_out_mem_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_sgd
