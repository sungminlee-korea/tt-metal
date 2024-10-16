# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv


class Down4:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(torch_model, "down4.conv1", [1, 40, 40, 256], (2, 2, 1, 1), reshard=True)
        self.conv2 = Conv(torch_model, "down4.conv2", [1, 20, 20, 512], (1, 1, 0, 0), deallocate=False)
        self.conv3 = Conv(torch_model, "down4.conv3", [1, 20, 20, 512], (1, 1, 0, 0))

        self.res1_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.0.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            height_sharding=False,
            deallocate=False,
        )
        self.res1_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.0.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res2_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.1.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res2_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.1.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res3_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.2.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res3_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.2.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res4_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.3.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res4_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.3.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res5_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.4.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res5_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.4.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res6_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.5.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res6_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.5.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res7_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.6.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res7_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.6.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.res8_conv1 = Conv(
            torch_model,
            "down4.resblock.module_list.7.0",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )
        self.res8_conv2 = Conv(
            torch_model,
            "down4.resblock.module_list.7.1",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )

        self.conv4 = Conv(
            torch_model,
            "down4.conv4",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
        )

        self.conv5 = Conv(
            torch_model,
            "down4.conv5",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
        )

    def __call__(self, device, input_tensor):
        output_tensor_split = self.conv1(device, input_tensor)
        output_tensor_split = ttnn.mish(output_tensor_split)
        output_tensor_left = self.conv2(device, output_tensor_split)
        output_tensor_left = ttnn.mish(output_tensor_left)

        res1_split = self.conv3(device, output_tensor_split)
        res1_split = ttnn.mish(res1_split)

        output_tensor = self.res1_conv1(device, res1_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res1_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(device, res2_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res2_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res3_split = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.res3_conv1(device, res3_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res3_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res4_split = res3_split + output_tensor

        ttnn.deallocate(res3_split)

        output_tensor = self.res4_conv1(device, res4_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res4_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res5_split = res4_split + output_tensor

        ttnn.deallocate(res4_split)

        output_tensor = self.res5_conv1(device, res5_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res5_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res6_split = res5_split + output_tensor

        ttnn.deallocate(res5_split)

        output_tensor = self.res6_conv1(device, res6_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res6_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res7_split = res6_split + output_tensor

        ttnn.deallocate(res6_split)

        output_tensor = self.res7_conv1(device, res7_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res7_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res8_split = res7_split + output_tensor

        ttnn.deallocate(res7_split)

        output_tensor = self.res8_conv1(device, res8_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res8_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = res8_split + output_tensor

        ttnn.deallocate(res8_split)

        output_tensor = self.conv4(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_left = ttnn.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv5(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        return output_tensor

    def __str__(self) -> str:
        this_str = ""
        index = 1
        for conv in self.convs:
            this_str += str(index) + " " + str(conv)
            this_str += " \n"
            index += 1
        return this_str
