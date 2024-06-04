# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded, TtRMSNorm
from ttnn import ReplicateTensorToMesh
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import LightweightModule


class TtTransformer(LightweightModule):
    def __init__(
        self,
        device_mesh,
        state_dict,
        args,
        dtype,
        layers,
        mode="decode",
        transformation_mats=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.device_mesh = device_mesh
        self.model_config = args.get_model_config()
        assert self.vocab_size > 0

        self.layers = [
            TtTransformerBlock(
                device_mesh=device_mesh,
                state_dict=state_dict,
                args=args,
                dtype=dtype,
                layer_num=i,
                mode=mode,
                transformation_mats=transformation_mats,
            )
            for i in layers
        ]
        self.norm = TtRMSNorm(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=None,
            weight_key="norm",
            mode=mode,
        )

        self.state_dict = state_dict

        if args.dummy_weights:
            output_cache_name = None
        else:
            output_cache_name = args.weight_cache_path(dtype) / "output_multidevice_4d.weight"

        self.output_weight = ttnn.as_tensor(
            self.state_dict["output.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0),
            device=device_mesh,
            layout=self.model_config["OUTPUT_W_LAYOUT_TILE"],
            dtype=dtype,
            memory_config=self.model_config["OUTPUT_WEIGHTS_MEMCFG"],
            cache_file_name=output_cache_name,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        self.compute_kernel = self.args.get_compute_kernel_config()
        self.mode = mode

    def forward(
        self,
        x,
        start_pos,
        current_pos,
        attn_masks,
        rot_mats,
    ):
        seqlen = x.shape[2]
        for i, layer in enumerate(self.layers):
            if self.mode == "prefill":
                x = layer.forward_prefill(x, start_pos, current_pos, attn_masks, rot_mats)
            else:
                x = layer(x, start_pos, current_pos, attn_masks, rot_mats)
        attn_masks.deallocate(True)

        x_norm = self.norm(x)
        if self.mode == "prefill":
            matmul_prg_cfg = self.model_config["OUTPUT_MM_PROGCFG_PREFILL"](seqlen)
        else:
            matmul_prg_cfg = self.model_config["OUTPUT_MM_PROGCFG"]
        outputs = ttnn.matmul(
            x_norm,
            self.output_weight,
            # compute_with_storage_grid_size=(8, 8),
            # program_config=matmul_prg_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.experimental.tensor.WormholeComputeKernelConfig(
                math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        return outputs
