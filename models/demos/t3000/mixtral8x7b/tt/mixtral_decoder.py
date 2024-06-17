# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_mlp import TtMixtralMLP
from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded, TtRMSNorm
from models.demos.t3000.mixtral8x7b.tt.mixtral_moe import TtMoeLayer
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import LightweightModule


class TtTransformerBlock(LightweightModule):
    def __init__(
        self,
        device_mesh,
        state_dict,
        args,
        layer_num,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh

        self.args = args

        self.layer_num = layer_num
        self.attention = TtMixtralAttention(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )

        self.feed_forward = TtMoeLayer(
            device_mesh=device_mesh,
            state_dict=state_dict,
            experts=TtMixtralMLP(
                device_mesh=device_mesh,
                state_dict=state_dict,
                args=args,
                layer_num=layer_num,
                dtypes={
                    "w1": ttnn.bfloat4_b,
                    "w2": ttnn.bfloat8_b,
                    "w3": ttnn.bfloat4_b,
                },
            ),
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.attention_norm = TtRMSNorm(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=layer_num,
            weight_key="attention_norm",
        )

        self.ffn_norm = TtRMSNorm(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=layer_num,
            weight_key="ffn_norm",
        )
        self.comps = []

    def forward(
        self, xs_1SBH, start_pos, current_pos, attn_masks, rot_mats, transformation_mats=None, user_id=0, mode="decode"
    ) -> ttnn.Tensor:
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B: batch dim (32)
        S: seq dim (1)
        1: unary dim
        H: hidden dim (4096)
        """
        attn_norm_1SBH = self.attention_norm(xs_1SBH)
        # self.comps.append(
        #     ttnn.to_torch(attn_norm_1SBH, mesh_composer=ttnn.ConcatMeshToTensor(self.device_mesh, dim=0))[0]
        # )

        attn_1SBH = self.attention(
            attn_norm_1SBH,
            start_pos,
            current_pos,
            attn_masks,
            rot_mats,
            transformation_mats,
            user_id,
            mode,
        )
        # attn_norm_1SBH.deallocate(True)
        # self.comps.append(ttnn.to_torch(attn_1SBH, mesh_composer=ttnn.ConcatMeshToTensor(self.device_mesh, dim=0))[0])

        hs_1SBH = ttnn.experimental.tensor.add(xs_1SBH, attn_1SBH, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)
        # self.comps.append(ttnn.to_torch(hs_1SBH, mesh_composer=ttnn.ConcatMeshToTensor(self.device_mesh, dim=0))[0])

        ffn_norm_1SBH = self.ffn_norm(hs_1SBH)
        # hs_1SBH.deallocate(True)

        # moe_input = ttnn.to_torch(ffn_norm_1SBH, mesh_composer=ttnn.ConcatMeshToTensor(self.device_mesh, dim=0))[0]
        # self.comps.append(moe_input)
        # torch.save(moe_input, "moe_input.pt")

        ffn_1SBH = self.feed_forward(ffn_norm_1SBH, mode=mode)
        # ffn_norm_1SBH.deallocate(True)
        # self.comps.append(ttnn.to_torch(ffn_1SBH, mesh_composer=ttnn.ConcatMeshToTensor(self.device_mesh, dim=0))[0])

        out_1SBH = ttnn.add(hs_1SBH, ffn_1SBH, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # self.comps.append(ttnn.to_torch(out_1SBH, mesh_composer=ttnn.ConcatMeshToTensor(self.device_mesh, dim=0))[0])
        print("layer num =", self.layer_num)
        return out_1SBH
