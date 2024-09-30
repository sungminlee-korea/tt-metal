import ttnn


class VGG_TTNN:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,
        final_output_mem_config,
        num_classes=1000,
        dealloc_input=True,
    ):
        self.device = device
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
        # conv1
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        conv_config = ttnn.Conv2dConfig(
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            math_fidelity=self.model_config["MATH_FIDELITY"],
            shard_layout=ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            input_channels_alignment=16,
            # deallocate_activation=self.dealloc_input,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            act_block_h_override=64,
            # activation="relu"
        )
        for d in dir(conv_config):
            if d.startswith("__"):
                continue
            print(f"{d}:{getattr(conv_config,d)}")
        # breakpoint()
        [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
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
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        breakpoint()
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.classifier(x)
        return x


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
