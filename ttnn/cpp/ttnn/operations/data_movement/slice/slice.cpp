// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "slice.hpp"
#include "device/slice_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/copy/copy.hpp"


namespace ttnn::operations::data_movement {
namespace detail {
    static inline uint32_t wrap_index(int index, int size) {
        return index < 0 ? size + index : index;
    }
    static inline uint32_t round_up_to_multiple_of_32(uint32_t value) {
        return value == 0 ? 32u : ((value + 31u) & ~31);
    }
}

template<typename T>
ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::vector<T> &begins,
    const std::vector<T> &ends,
    const std::vector<T> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {

    // Ensure start and end vectors have matching sizes and correct tensor rank
    uint32_t input_rank = input_tensor.get_shape().rank();
    const auto &input_shape = input_tensor.get_shape();
    bool no_step = std::ranges::all_of(step, [](uint32_t s) { return s == 1; });
    bool starts_zero = std::ranges::all_of(begins, [](uint32_t s) { return s == 0; });
    bool ends_max = true;
    for (size_t i = 0; i < ends.size(); ++i) {
        ends_max &= ends[i] == input_shape[i];
        if (!ends_max) {
            break;
        }
    }
    if (no_step && starts_zero && ends_max) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
            return ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
        }
        return input_tensor;
    }

    TT_FATAL(input_rank == begins.size(), "Input rank {} and begins {} must have the same size", input_rank, begins.size());
    TT_FATAL(begins.size() == ends.size(), "Start {} and end {} must have the same size", begins.size(), ends.size());
    TT_FATAL(step.size() == begins.size(), "Step {} must have the same size as start {} and end", step.size(), begins.size());

    // Create modified vectors with appropriate size (max rank 4) and wrap indices
    Tensor input_4d = (input_rank < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto padded_4d_shape = input_4d.get_legacy_shape();
    std::array<uint32_t, 4> modified_begins = {0, 0, 0, 0};
    std::array<uint32_t, 4> modified_ends = {padded_4d_shape[0], padded_4d_shape[1], padded_4d_shape[2], padded_4d_shape[3]};
    std::array<uint32_t, 4> modified_step = {1, 1, 1, 1};
    uint32_t rank_diff = 4 - input_rank;

    // Ideally we would call the 4D array implementation of slice here and then handle reshapes and padding outside of it but it's not ready yet
    // Insert values for start, step and end, wrapping indices using detail::wrap_index
    // should be able to skip wrap_index if T is uint32_t
    for (size_t i = 0; i < begins.size(); ++i) {
        modified_begins[i + rank_diff] = detail::wrap_index(begins[i], input_tensor.get_shape()[i]);
        modified_ends[i + rank_diff] = detail::wrap_index(ends[i], input_tensor.get_shape()[i]);
        modified_step[i + rank_diff] = step[i];
    }

    auto output_dim_i = [&modified_begins, &modified_step] (size_t i, const std::array<uint32_t, 4> &modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    std::array<uint32_t, 4> padded_ends = modified_ends;
    if (input_tensor.layout() == Layout::TILE) {
        padded_ends[2] = detail::round_up_to_multiple_of_32(padded_ends[2]);
        padded_ends[3] = detail::round_up_to_multiple_of_32(padded_ends[3]);
    }
    std::vector<uint32_t> actual_shape, padded_shape;
    actual_shape.reserve(input_rank);
    padded_shape.reserve(input_rank);
    bool empty = false;
    for (int i = 0; i < input_rank; ++i) {
        // Check that end indices are greater than or equal to start indices (empty tensor where end=start is supported)
        TT_FATAL(modified_ends[i + rank_diff] >= modified_begins[i + rank_diff], "End {} must be greater than or equal to start {}", modified_ends[i + rank_diff], modified_begins[i + rank_diff]);
        auto val = output_dim_i(i + rank_diff, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape.push_back(val);
        padded_shape.push_back(std::max(output_dim_i(i + rank_diff, padded_ends), (uint32_t)1));
    }

    ttnn::Shape output_shape(actual_shape, padded_shape);
    // PyTorch supports final dimension = 0 (start = end, where end is inclusive) so >= is okay, just return an empty tensor
    if (empty) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Host tensor slice cannot return a scalar or empty tensor");
        return ttnn::empty(output_shape, input_tensor.dtype(), input_tensor.layout(),
            input_tensor.device(), memory_config_arg.value_or(input_tensor.memory_config()));
    }

    // Early exit if slice is a no-op (ends = padding ends and step = 1 for all dimensions)
    if (tt::tt_metal::LegacyShape(padded_shape) == input_tensor.get_legacy_shape() and no_step) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
            auto res = ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
            return ttnn::reshape(res, output_shape);
        }
        return ttnn::reshape(input_tensor, output_shape);
    }

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        TT_FATAL(no_step, "Host tensor slice does not support strides");
        // if we support negative strides, we can't do this early exit
        if (input_tensor.get_legacy_shape() == actual_shape) {
            return input_tensor;
        } else {
            auto input_4d_rm = ttnn::to_layout(input_4d, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device *)nullptr);
            auto output_4d =  input_4d_rm.unpad(tt::tt_metal::LegacyShape(modified_begins), tt::tt_metal::LegacyShape(modified_ends));
            auto output_4d_rm = ttnn::to_layout(output_4d, input_tensor.get_layout(), std::nullopt, std::nullopt, (Device *)nullptr);
            return ttnn::reshape(output_4d_rm, output_shape);
        }
    }
    else {
        // TODO: Generalize this early exit of slice for other cases
        auto& input_tensor_shape = input_4d.get_legacy_shape();
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
        if (input_4d.is_sharded() && input_4d.memory_config() == memory_config &&
            input_tensor_shape.rank() > 1) {
            TT_FATAL(no_step, "Sharded tensor slice implementation does not support striding");
            uint32_t i;
            // Require all leading dims to be 1 (TODO: This can be relaxed to support outermost non-1 dim unpadding)
            bool in_place_unpad = true;
            for (i = 0; i < input_4d.get_legacy_shape().rank() - 2; ++i) {
                in_place_unpad &=
                    modified_begins[i] == 0 && modified_ends[i] == 1 && input_tensor_shape[i] == 1;
            }
            in_place_unpad &= modified_begins[i] == 0 &&
                              tt::div_up(modified_ends[i], input_4d.shard_spec().value().shape[0]) ==
                                  tt::div_up(input_tensor_shape[i], input_4d.shard_spec().value().shape[0]);
            i++;
            in_place_unpad &= modified_begins[i] == 0 && modified_ends[i] == input_tensor_shape[i];
            if (in_place_unpad) {
                return ttnn::reshape(input_tensor, output_shape);
            }
        }

        auto res = operation::run(
                   SliceDeviceOperation{
                    tt::tt_metal::LegacyShape(modified_begins),
                    tt::tt_metal::LegacyShape(padded_ends),
                    modified_step,
                    memory_config},
                    {input_4d}, {}, {optional_output_tensor}, queue_id)
            .at(0);
        return ttnn::reshape(res, output_shape);
    }
}
template<typename T>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::vector<T> &begins,
    const std::vector<T> &ends,
    const std::vector<T> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
        return SliceOperation::invoke<T>(ttnn::DefaultQueueId, input_tensor, begins, ends, step, memory_config_arg);
    }

// Specialization for uint32_t and N=4
template<>
ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4> &begins,
    const std::array<uint32_t, 4> &ends,
    const std::array<uint32_t, 4> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {

    const auto& padded_input_shape = input_tensor.get_shape().with_tile_padding();
    TT_FATAL(padded_input_shape.rank() == 4, "Input tensor must have rank 4");

    bool no_step = step[0] == 1 && step[1] == 1 && step[2] == 1 && step[3] == 1;
    bool starts_zero = begins[0]==0 && begins[1]==0 && begins[2]==0 && begins[3]==0;
    bool ends_max = ends[0]==padded_input_shape[0] && ends[1]==padded_input_shape[1] && ends[2]==padded_input_shape[2] && ends[3]==padded_input_shape[3];

    if (no_step && starts_zero && ends_max) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
            return ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
        }
        return input_tensor;
    }

    const bool tiled = input_tensor.get_layout() == Layout::TILE;
    bool on_device = input_tensor.storage_type() == StorageType::DEVICE;

    std::array<uint32_t, 4> actual_shape;
    std::array<uint32_t, 4> padded_shape;
    const std::array<uint32_t, 4> padded_ends = tiled ? std::array<uint32_t, 4>({ends[0], ends[1], detail::round_up_to_multiple_of_32(ends[2]), detail::round_up_to_multiple_of_32(ends[3])}) : ends;
    bool empty = false;
    for (int i = 0; i < 4; ++i) {
        TT_FATAL(ends[i] >= begins[i], "End {} must be greater than or equal to start {}", ends[i], begins[i]);
        uint32_t offset = step[i] - begins[i] - 1;
        uint32_t dim_size = (ends[i] + offset) / step[i];
        empty |= dim_size == 0;
        actual_shape[i] = dim_size;
        padded_shape[i]= std::max((padded_ends[i] + offset) / step[i], 1u);
    }

    ttnn::Shape output_shape(actual_shape, padded_shape);

    if (empty) {
        TT_FATAL(on_device, "Host tensor slice cannot return a scalar or empty tensor");
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
        return ttnn::empty(output_shape, input_tensor.dtype(), input_tensor.layout(),
            input_tensor.device(), memory_config);
    }

    // Early exit if slice is a no-op
    if (ttnn::Shape(padded_shape) == padded_input_shape && no_step) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
            auto res = ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
            return ttnn::reshape(res, output_shape);
        }
        return ttnn::reshape(input_tensor, output_shape); // change to view
    }

    if (on_device) {
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());

        // Check for in-place unpad optimization
        if (input_tensor.is_sharded() && input_tensor.memory_config() == memory_config && padded_input_shape.rank() > 1) {
            TT_FATAL(no_step, "Sharded tensor slice implementation does not support striding");
            bool in_place_unpad = true;
            for (int i = 0; i < 2; ++i) {
                in_place_unpad &= begins[i] == 0 && ends[i] == 1 && padded_input_shape[i] == 1;
            }
            in_place_unpad &= begins[2] == 0 &&
                                tt::div_up(ends[2], input_tensor.shard_spec().value().shape[0]) ==
                                    tt::div_up(padded_input_shape[2], input_tensor.shard_spec().value().shape[0]);
            in_place_unpad &= begins[3] == 0 && ends[3] == padded_input_shape[3];
            if (in_place_unpad) {
                return ttnn::reshape(input_tensor, output_shape);
            }
        }

        auto res = operation::run(
                    SliceDeviceOperation{
                    begins,
                    padded_ends,
                    step,
                    memory_config},
                    {input_tensor}, {}, {optional_output_tensor}, queue_id)[0];
        return ttnn::reshape(res, output_shape);
    }

    TT_FATAL(no_step, "Host tensor slice does not support strides");

    if (input_tensor.get_legacy_shape() == actual_shape) {
        return input_tensor;
    } else {
        auto input_4d_rm = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device *)nullptr);
        auto output_4d =  input_4d_rm.unpad(tt::tt_metal::LegacyShape(begins), tt::tt_metal::LegacyShape(ends));
        auto output_4d_rm = ttnn::to_layout(output_4d, input_tensor.get_layout(), std::nullopt, std::nullopt, (Device *)nullptr);
        return ttnn::reshape(output_4d_rm, output_shape);
    }
}

template<typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<T, N> &output_tensor_start,
    const std::array<T, N> &output_tensor_end,
    const std::array<T, N> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
        std::vector<int> start(output_tensor_start.begin(), output_tensor_start.end());
        std::vector<int> end(output_tensor_end.begin(), output_tensor_end.end());
        std::vector<int> step_vec(step.begin(), step.end());
        return SliceOperation::invoke<int>(queue_id, input_tensor, start, end, step_vec, memory_config_arg);
    }

template<typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::array<T, N> &output_tensor_start,
    const std::array<T, N> &output_tensor_end,
    const std::array<T, N> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
        return SliceOperation::invoke<T, N>(ttnn::DefaultQueueId, input_tensor, output_tensor_start, output_tensor_end, step, memory_config_arg);
    }


template ttnn::Tensor SliceOperation::invoke<int>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::vector<int> &begins,
    const std::vector<int> &ends,
    const std::vector<int> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<int>(
    const ttnn::Tensor& input_tensor,
    const std::vector<int> &begins,
    const std::vector<int> &ends,
    const std::vector<int> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);


template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::vector<uint32_t> &begins,
    const std::vector<uint32_t> &ends,
    const std::vector<uint32_t> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    const ttnn::Tensor& input_tensor,
    const std::vector<uint32_t> &begins,
    const std::vector<uint32_t> &ends,
    const std::vector<uint32_t> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4> &output_tensor_start,
    const std::array<uint32_t, 4> &output_tensor_end,
    const std::array<uint32_t, 4> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 1> &output_tensor_start,
    const std::array<uint32_t, 1> &output_tensor_end,
    const std::array<uint32_t, 1> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 1> &output_tensor_start,
    const std::array<uint32_t, 1> &output_tensor_end,
    const std::array<uint32_t, 1> &step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

}  // namespace operations
