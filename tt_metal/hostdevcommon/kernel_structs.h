// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace tt {

// All CBs can used for dataflow in/out
// Certain CBs are specifically designed to handle compute input, output, and intermediates.
enum CB : std::uint8_t
{
  // Designed to be used as compute inputs, or dataflow in/out
  cb_0       = 0,
  cb_1       = 1,
  cb_2       = 2,
  cb_3       = 3,
  cb_4       = 4,
  cb_5       = 5,
  cb_6       = 6,
  cb_7       = 7,

  // Dataflow in/out only
  cb_8           = 8,
  cb_9           = 9,
  cb_10           = 10,
  cb_11           = 11,
  cb_12           = 12,
  cb_13           = 13,
  cb_14           = 14,
  cb_15           = 15,

  // Designed to be used as compute outputs, or dataflow in/out
  cb_16      = 16,
  cb_17      = 17,
  cb_18      = 18,
  cb_19      = 19,
  cb_20      = 20,
  cb_21      = 21,
  cb_22      = 22,
  cb_23      = 23,

  // Designed to be used as compute intermediates, or dataflow in/out
  cb_24 = 24,
  cb_25 = 25,
  cb_26 = 26,
  cb_27 = 27,
  cb_28 = 28,
  cb_29 = 29,
  cb_30 = 30,
  cb_31 = 31,
};
  /////////////////////////////
 // end of user facing APIs //
/////////////////////////////


enum class Dim : std::uint8_t
{
    None      = 0,
    R         = 1,
    C         = 2,
    Z         = 3,
    RC        = 4,
    ZR        = 5,
    Invalid   = 0xFF,
};

enum class SfpuOp : std::uint8_t
{
    Exp,
    Log,
    Sqrt,
    Gelu,
    GeluDerivative,
    Reciprocal,
    Sigmoid,
    Tanh,
    Dropout,
    Datacopy, // This just means passthrough and no sfpu
    Transpose, // datacopy + math transpose
    Invalid,
};

enum class BinaryOp : std::uint8_t
{
    Add,
    Subtract,
    Multiply,
    Power,
    AddRelu, // Add + Relu
    SubtractRelu, // Subtract + Relu
    Invalid,
};

enum class TmOp : std::uint8_t
{
    rBroadcast,
    cBroadcast,
    zBroadcast,
    hSlice,
    hStack,
    vSlice,
    vStack,
    Transpose,
    Invalid,
};

enum class ReduceFunc : std::uint8_t
{
    Sum,
    Avg,  // Needed only on tensor level to compute correct coefficient. Kernel uses Sum.
    Max,
    Invalid,
};

enum DstMode : std::uint8_t
{
    Full          = 0,
    Half          = 1,
    Tile          = 2,
    NUM_DST_MODES = 3,
};

enum class Action {
    None,
    Slice,
    Stack,
};

// To be deprecated: the old enum from which CBs evolved
enum HlkOperand : std::uint8_t
{
  in0       = 0,
  in1       = 1,
  in2       = 2,
  in3       = 3,
  in4       = 4,
  in5       = 5,
  in6       = 6,
  in7       = 7,

  param0    = 8,
  param1    = 9,
  param2    = 10,
  param3    = 11,
  param4    = 12,
  param5    = 13,
  param6    = 14,
  param7    = 15,

  out0      = 16,
  out1      = 17,
  out2      = 18,
  out3      = 19,
  out4      = 20,
  out5      = 21,
  out6      = 22,
  out7      = 23,

  intermed0 = 24,
  intermed1 = 25,
  intermed2 = 26,
  intermed3 = 27,
  intermed4 = 28,
  intermed5 = 29,
  intermed6 = 30,
  intermed7 = 31,
};

enum class OpCode : uint8_t
{
    Exponential = 0,
    Reciprocal  = 1,
    Gelu        = 2,
    Add         = 3,
    Subtract    = 4,
    Multiply    = 5

};

enum class PullAndPushConfig : uint8_t {
    LOCAL = 0,                  // fast dispatch only on local chip
    PUSH_TO_REMOTE = 1,         // read from issue queue and write data to CB on SRC router on issue path
    REMOTE_PULL_AND_PUSH = 2,   // read from CB on DST router on issue path and push to CB on SRC router on completion path
    PULL_FROM_REMOTE = 3        // read from CB on DST router on completion path and write to completion queue
};

constexpr std::uint32_t NUM_MAX_IN_BUFFERS_PER_CORE = HlkOperand::in7 - HlkOperand::in0 + 1;
constexpr std::uint32_t NUM_MAX_PARAM_BUFFERS_PER_CORE = HlkOperand::param7 - HlkOperand::param0 + 1;
constexpr std::uint32_t NUM_MAX_OUT_BUFFERS_PER_CORE = HlkOperand::out7 - HlkOperand::out0 + 1;
constexpr std::uint32_t NUM_MAX_INTERMED_BUFFERS_PER_CORE = HlkOperand::intermed7 - HlkOperand::intermed0 + 1;

} //  namespace tt
