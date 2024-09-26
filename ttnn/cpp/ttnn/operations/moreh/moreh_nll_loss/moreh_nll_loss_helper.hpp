// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

typedef const std::string loss_reduction;

namespace ttnn::operations::moreh {

loss_reduction NONE = "none";
loss_reduction SUM = "sum";
loss_reduction MEAN = "mean";

}  // namespace ttnn::operations::moreh
