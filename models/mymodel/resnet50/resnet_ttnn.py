import ttnn
import math


def _nearest_32(x):
    return math.ceil(x / 32) * 32


class RESNET_TTNN:
    def __init__(
        self,
        device,
        parameters,
        torch_input,
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

    def __call__(self, x):
        """
        conv1
        Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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
