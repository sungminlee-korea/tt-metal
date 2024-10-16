// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

enum class DataMovementProcessor {
    RISCV_0 = 0,  // DM0
    RISCV_1 = 1,  // DM1
};

enum NOC : uint8_t {
    RISCV_0_default = 0,
    RISCV_1_default = 1,
    NOC_0 = 0,
    NOC_1 = 1,
};

enum NOC_MODE : uint8_t {
    DM_DEDICATED_NOC = 0,
    DM_DYNAMIC_NOC = 1,
};

enum Eth : uint8_t {
    ACTIVE = 0,
    IDLE = 2,
};

} // namespace tt::tt_metal
