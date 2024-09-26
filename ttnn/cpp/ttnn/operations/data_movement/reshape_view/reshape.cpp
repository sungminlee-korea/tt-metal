// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "tt_metal/common/constants.hpp"
#include <ttnn/operations/numpy/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {


namespace detail {

    ttnn::Tensor host_row_major_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
        if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
            auto device = tensor.device();
            auto memory_config = tensor.memory_config();
            auto host_tensor = ttnn::data_transfer_to_host(tensor);
            auto layout = tensor.get_layout();
            auto host_reshape_tensor = host_tensor.reshape(shape.value);
            return ttnn::data_transfer_to_device(host_reshape_tensor, device, memory_config);
        }
        else{
            return tensor.reshape(shape.value);
        }
    }


    ttnn::Tensor row_major_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {

            const auto layout = tensor.get_layout();
            auto tensor_shape = tensor.get_shape();
            if (tensor.is_contiguous()) {
                if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
                    // Page size depends on the width, so only modify the shape if the width is the same
                    if (tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]) {
                        return tensor.reshape(shape.value);
                    }
                    //Different page width, need to remake tensor
                    else {
                        uint32_t ROW_MAJOR_WIDTH = 8;
                        auto original_rank = shape.rank();

                        bool host_reshape = !(tensor_shape[-1] % ROW_MAJOR_WIDTH == 0 && shape[-1] % ROW_MAJOR_WIDTH == 0);
                        if(host_reshape) {
                            return  host_row_major_reshape(tensor, shape);
                        }
                        else {
                            auto tensor_4d = unsqueeze_to_4D(tensor);
                            const auto shape_4d = shape.to_rank<4>();
                            auto rm_tensor = ttnn::to_layout(tensor_4d, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
                            auto unsqueezed_tensor = ttnn::reshape_on_device(rm_tensor, shape_4d[0], shape_4d[1], shape_4d[2], shape_4d[3], tensor.memory_config());
                            auto squeezed_tensor = squeeze_from_4D(unsqueezed_tensor, original_rank);
                            return ttnn::to_layout(squeezed_tensor, layout, std::nullopt, std::nullopt, (Device *)nullptr);
                        }
                    }
                } else {
                    return tensor.reshape(shape.value);
                }
            } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
                // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
                // is possible
                if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                    tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1] and
                    tensor_shape.with_tile_padding()[-2] == shape.with_tile_padding()[-2]) {
                    return tensor.reshape(shape.value);
                }
            }
            return host_row_major_reshape(tensor, shape);
    }


}


ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape
    ) {

    auto tensor_shape = tensor.get_shape();
    if (tensor_shape == shape) {
        return {tensor};
    }

    const auto layout = tensor.get_layout();

    if (layout == ttnn::Layout::ROW_MAJOR) {
        return detail::row_major_reshape(tensor, shape);
    } else {
        const auto new_shape_with_tile_padding = shape.with_tile_padding();
        const auto new_height = new_shape_with_tile_padding[-2];
        const auto new_width = new_shape_with_tile_padding[-1];

        const auto is_tile_multiple = (new_height % ttnn::TILE_SIZE == 0 && new_width % ttnn::TILE_SIZE == 0);
        if (not is_tile_multiple) {
            return detail::row_major_reshape(tensor, shape);
        }

        if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
            if (tensor_shape.with_tile_padding()[-1] == new_width) {
                return tensor.reshape(shape.value);
            }
        } else {
            return tensor.reshape(shape.value);
        }
    }
    return detail::row_major_reshape(tensor, shape);


}



} // ttnn::operations::data_movement namespace
