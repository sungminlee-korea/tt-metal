import ttnn
import math


def _nearest_32(x):
    return math.ceil(x / 32) * 32


class VGG_TTNN:
    def __init__(
        self,
        device,
        parameters,
        torch_model,
        torch_input,
        batch_size,
        model_config,
        input_shape,
        final_output_mem_config,
        num_classes=1000,
        dealloc_input=True,
    ):
        self.device = device
        self.torch_model = (torch_model,)
        self.torch_input = (torch_input,)
        self.parameters = parameters
        self.batch_size = batch_size
        self.model_config = model_config
        self.input_shape = input_shape
        self.final_output_mem_config = final_output_mem_config
        self.num_classes = num_classes
        self.dealloc_input = dealloc_input

        self.device = device
        self.conv1 = _conv2d()
        self.conv2 = _conv2d()
        self.conv3 = _conv2d()
        self.conv4 = _conv2d()
        self.conv5 = _conv2d()
        self.classifier = _classifier()

    def __call__(self, x):
        """
        conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        conv1_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            input_channels_alignment=16,  # 기본값 32.. in_c 3 이하면 16
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=64,
            activation="relu",
            output_layout=ttnn.Layout.TILE,
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv1.0.weight"]),
            in_channels=3,
            out_channels=64,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv1.0.bias"]), [1, 1, 1, 64]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=224,
            input_width=224,
            conv_config=conv1_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        x = self._maxpool2d(x, self.batch_size, out_height, out_width, 64, 2, 2, 0, 1)
        """
        conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        conv2_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            # input_channels_alignment=64,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=64,
            activation="relu",
            output_layout=ttnn.Layout.TILE,
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv2.0.weight"]),
            in_channels=64,
            out_channels=128,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv2.0.bias"]), [1, 1, 1, 128]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=112,
            input_width=112,
            conv_config=conv2_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        x = self._maxpool2d(x, self.batch_size, out_height, out_width, 128, 2, 2, 0, 1)
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        # 1st conv in block3
        conv30_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            # input_channels_alignment=128,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=64,
            activation="relu",
            output_layout=ttnn.Layout.ROW_MAJOR,
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv3.0.weight"]),
            in_channels=128,
            out_channels=256,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv3.0.bias"]), [1, 1, 1, 256]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=56,
            input_width=56,
            conv_config=conv30_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )

        # 2nd conv in block3
        conv31_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            # input_channels_alignment=32,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=32,
            activation="relu",
            output_layout=ttnn.Layout.TILE,
        )

        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv3.2.weight"]),
            in_channels=256,
            out_channels=256,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv3.2.bias"]), [1, 1, 1, 256]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=56,
            input_width=56,
            conv_config=conv31_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        x = self._maxpool2d(x, self.batch_size, out_height, out_width, 256, 2, 2, 0, 1)
        """
        h,w : 28
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        # 1st conv in block4
        conv40_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=32,
            activation="relu",
            reshard_if_not_optimal=True,
            output_layout=ttnn.Layout.ROW_MAJOR,
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv4.0.weight"]),
            in_channels=256,
            out_channels=512,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv4.0.bias"]), [1, 1, 1, 512]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=28,
            input_width=28,
            conv_config=conv40_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        config_to_use = x.memory_config()
        # 2nd conv in block4
        conv41_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            # input_channels_alignment=32,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=32,
            activation="relu",
            reshard_if_not_optimal=True,
            # output_layout=ttnn.Layout.ROW_MAJOR
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv4.2.weight"]),
            in_channels=512,
            out_channels=512,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv4.2.bias"]), [1, 1, 1, 512]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=28,
            input_width=28,
            conv_config=conv41_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        x = ttnn.to_memory_config(x, config_to_use)
        x = self._maxpool2d(x, self.batch_size, out_height, out_width, 512, 2, 2, 0, 1)
        config_to_use = x.memory_config()
        """
        h,w : 14
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        # 1st conv in block5
        conv50_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=32,
            activation="relu",
            reshard_if_not_optimal=True,
            # output_layout=ttnn.Layout.ROW_MAJOR
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv5.0.weight"]),
            in_channels=512,
            out_channels=512,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv5.0.bias"]), [1, 1, 1, 512]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=14,
            input_width=14,
            conv_config=conv50_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        # 2nd conv in block5
        conv51_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            # input_channels_alignment=32,
            deallocate_activation=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=32,
            activation="relu",
            reshard_if_not_optimal=True,
            # output_layout=ttnn.Layout.ROW_MAJOR
        )
        x, out_height, out_width, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.from_torch(self.parameters["conv5.2.weight"]),
            in_channels=512,
            out_channels=512,
            device=self.device,
            bias_tensor=ttnn.reshape(ttnn.from_torch(self.parameters["conv5.2.bias"]), [1, 1, 1, 512]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            batch_size=self.batch_size,
            input_height=14,
            input_width=14,
            conv_config=conv51_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)  # drop bracket
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # drop bracket
        x = ttnn.to_memory_config(x, config_to_use)
        x = self._maxpool2d(x, self.batch_size, out_height, out_width, 512, 2, 2, 0, 1)
        # enf of conv blocks
        """
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        """
        # x fallback to host (channel changing not supported)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.from_device(x)
        x = ttnn.reshape(x, (2, 1, 1, 512 * 49))
        x = ttnn.to_device(x, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        # 1 linear
        weights = ttnn.from_torch(
            self.parameters["classifier.1.weight"].permute(1, 0), layout=ttnn.TILE_LAYOUT, device=self.device
        )
        bias = self.parameters["classifier.1.bias"].repeat(2, 1, 1, 1)
        bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=self.device)
        x = ttnn.relu(ttnn.matmul(x, weights, memory_config=ttnn.L1_MEMORY_CONFIG) + bias)
        # 2 linear
        weights = ttnn.from_torch(
            self.parameters["classifier.4.weight"].permute(1, 0), layout=ttnn.TILE_LAYOUT, device=self.device
        )
        bias = self.parameters["classifier.4.bias"].repeat(2, 1, 1, 1)
        bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=self.device)
        x = ttnn.relu(ttnn.matmul(x, weights, memory_config=ttnn.L1_MEMORY_CONFIG) + bias)

        # 3 linear
        weights = ttnn.from_torch(
            self.parameters["classifier.7.weight"].permute(1, 0), layout=ttnn.TILE_LAYOUT, device=self.device
        )
        bias = self.parameters["classifier.7.bias"].repeat(2, 1, 1, 1)
        bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=self.device)
        return x

    def _maxpool2d(self, x, batch_size, h, w, channels, kernel_size, stride, padding, dilation):
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=batch_size,
            input_h=h,
            input_w=w,
            channels=channels,
            kernel_size=[kernel_size, kernel_size],
            stride=[stride, stride],
            padding=[padding, padding],
            dilation=[dilation, dilation],
            device=self.device,
        )
        return x

    def _print_config(self, config):
        for d in dir(config):
            if d.startswith("__"):
                continue
            print(f"{d}:{getattr(config,d)}")


class _conv2d:
    def __init__(self):
        pass

    def __call__(self, x):
        pass
        return x


class _classifier:
    def __init__(self):
        pass

    def __call__(self, x):
        pass
        return x
