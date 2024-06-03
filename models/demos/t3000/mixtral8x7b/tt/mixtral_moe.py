# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import LightweightModule


class TtMoeLayer(LightweightModule):
    def __init__(self, device_mesh, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.device_mesh = device_mesh
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        gate_name = f"layers.{layer_num}.feed_forward.gate.weight"
        if args.dummy_weights:
            cache_name = None
        else:
            cache_name = args.weight_cache_path(dtype) / (gate_name + "_multidevice_repadded")

        self.gates_H8 = ttnn.as_tensor(
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (1, 55), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        self.num_devices = 8
        self.compute_kernel = args.get_compute_kernel_attn_config()

        self.expert_mask_11BB = ttnn.from_torch(
            torch.cat([torch.full((1, 1, 32, 32), fill_value=i + 1) for i in range(8)], dim=3),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
        )

        top8_mask = torch.full((1, 1, 1, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, 1:9] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.top8_mask_11B_64 = ttnn.sum(self.top8_mask_11B_64, dim=2)

        top2_mask = torch.full((1, 1, 1, 32), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.top2_mask_11BB = ttnn.sum(self.top2_mask_11BB, dim=2)

        reduce_mask_torch = torch.zeros(1, 1, self.args.max_seq_len, self.args.max_seq_len * 8)
        for i in range(self.args.max_seq_len):
            reduce_mask_torch[:, :, i, range(i, self.args.max_seq_len * 8, self.args.max_seq_len)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

        self.reduce_mask = ttnn.to_device(self.reduce_mask, self.device_mesh)

    def forward_prefill(self, inputs):
        """
        inputs: (seq_len, 1, batch, hidden_dim)

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len (1)
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        seqlen = input_i_1SBH.shape[2]
        # get logits for the experts
        gate_logits_1SB8 = ttnn.experimental.operations.primary.matmul(
            input_i_1SBH,
            self.gates_H8,
            program_config=self.model_config["GATE_MM_OUTPUT_PROGCFG"],
            output_mem_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
            # use_1d_systolic_array=True,
            # core_grid=ttnn.CoreGrid(y=4, x=8),
            output_dtype=ttnn.bfloat16,
        )

        # get weights for top-2 experts
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)

        # TODO: remove fallback to torch when top_k is ready
        # ttl_topk_values, ttl_topk_indices = ttnn.experimental.operations.primary.topk(gate_logits_1SB8, 32)
        gate_logits_1SB8 = ttnn.to_torch(gate_logits_1SB8, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[
            :1
        ]
        ttl_topk_values, ttl_topk_indices = torch.topk(gate_logits_1SB8, 32)
        ttl_topk_values = ttnn.from_torch(
            ttl_topk_values,
            device=self.device_mesh,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )
        ttl_topk_indices = ttnn.from_torch(
            ttl_topk_indices,
            device=self.device_mesh,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )

        # print("topk", ttl_topk_values, ttl_topk_indices)
        self.expert_mask_11BB = ttnn.experimental.tensor.repeat(self.expert_mask_11BB, [1, 1, int(seqlen / 32), 1])
        ttl_topk_values = ttnn.add(ttl_topk_values, self.top2_mask_11BB)
        mask_B2 = ttnn.eq(self.expert_mask_11BB, ttl_topk_indices)
        weights_1SB1 = ttnn.sum(ttnn.softmax(ttl_topk_values, dim=-1) * mask_B2, dim=3)
        # MLP and masking
        weights = expert_i_HH.forward_prefill(input_i_1SBH)
        results_11BH = ttnn.mul(weights, weights_1SB1)

        # all gather
        output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
        results_11BH.deallocate(True)
        # sum on each device
        # output_11BH_gathered = ttnn.experimental.tensor.reshape(output_11BH_gathered, 1, seqlen, 32, 4096)
        # output_11BH_gathered = ttnn.sum(output_11BH_gathered, dim=2)
        # output_11BH_gathered = ttnn.reshape(output_11BH_gathered, ttnn.Shape([1, 1, seqlen, 4096]))
        output_11BH_gathered = ttnn.experimental.operations.primary.matmul(self.reduce_mask, output_11BH_gathered)

        return output_11BH_gathered
