# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

#import tt_lib as ttl
import ttnn

from models.utility_functions import torch2tt_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        state_dict,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """
        '''
        x_proj_weight_name = "mixer.x_proj.weight"
        self.delta_t_proj_weights = torch2tt_tensor(
            self.state_dict[x_proj_weight_name][: self.args.dt_rank, :],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.delta_t_proj = Linear(self.args.d_inner, self.args.dt_rank, self.delta_t_proj_weights, bias=None)

        self.B_proj_weights = torch2tt_tensor(
            self.state_dict[x_proj_weight_name][self.args.dt_rank:(self.args.dt_rank+self.args.d_state), :],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )

        self.C_proj_weights = torch2tt_tensor(
            self.state_dict[x_proj_weight_name][(self.args.dt_rank+self.args.d_state):, :],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )

        self.B_proj = Linear(self.args.d_inner, self.args.d_state, self.B_proj_weights, bias=None)
        self.C_proj = Linear(self.args.d_inner, self.args.d_state, self.C_proj_weights, bias=None)

        A_weight_name = "mixer.A_log"
        self.A = self.state_dict[A_weight_name]
        self.A = -torch.exp(self.A.float())  # (2E, N)
        self.A = self.A.repeat(self.args.batch_size, 1, 1).reshape(
            self.args.batch_size, 1, -1, self.args.d_state
        )  # (BS, 1, 2E, N)

        self.A = torch2tt_tensor(
            self.A,
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )

        D_weight_name = "mixer.D"
        self.D = torch2tt_tensor(
            self.state_dict[D_weight_name].repeat(self.args.batch_size, 1).reshape(
                self.args.batch_size, 1, -1, self.args.d_inner
            ),
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.D = tt_lib.tensor.permute(self.D, [0, 2, 3, 1])

        dt_proj_weight_name = "mixer.dt_proj.weight"
        dt_proj_bias_name = "mixer.dt_proj.bias"
        self.dt_proj_weights = torch2tt_tensor(
            self.state_dict[dt_proj_weight_name],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.dt_proj_bias = torch2tt_tensor(
            self.state_dict[dt_proj_bias_name],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.dt_proj = Linear(self.args.dt_rank, self.args.d_inner, self.dt_proj_weights, bias=self.dt_proj_bias)
        '''
        
        self.tt_hidden_state = ttnn.zeros((1,1,32,40960), layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        '''
        self.tt_hidden_state = torch2tt_tensor(
            prev_hidden_states,
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        
        self.B_intermediate = torch.ones((args.batch_size, 1, args.d_inner, args.d_state))
        self.B_intermediate = torch2tt_tensor(
            self.B_intermediate,
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        '''
    def forward(self, x):
        '''
        delta_t = self.delta_t_proj(x)
        delta_t = self.dt_proj(delta_t)
        delta_t = tt_lib.tensor.softplus(
            delta_t,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
        )
        delta_t = tt_lib.tensor.permute(delta_t, [0, 2, 3, 1])

        B = self.B_proj(x)
        C = self.C_proj(x)

        B = tt_lib.tensor.transpose(B, 2, 1)
        C = tt_lib.tensor.permute(C, [0, 2, 3, 1])

        delta_A = tt_lib.tensor.bcast(
            self.A, delta_t, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )
        delta_A = tt_lib.tensor.exp(delta_A)

        delta_B = tt_lib.tensor.bcast(
            self.B_intermediate, B, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.H
        )
        B.deallocate()
        delta_B = tt_lib.tensor.bcast(
            delta_B, delta_t, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )

        delta_A_h = tt_lib.tensor.mul(delta_A, self.tt_hidden_state)

        x = tt_lib.tensor.permute(x, [0, 2, 3, 1])
        delta_B_x = tt_lib.tensor.bcast(
            delta_B, x, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )

        self.tt_hidden_state = tt_lib.tensor.add(delta_A_h, delta_B_x)
        self.output = tt_lib.tensor.bmm(self.tt_hidden_state, C)
        C.deallocate()
        x = tt_lib.tensor.mul(self.D, x)
        self.output = tt_lib.tensor.add(self.output, x)
        x.deallocate()

        self.output = tt_lib.tensor.permute(self.output, [0, 3, 1, 2])
        '''

        num_users = 32
        hidden_dim = 40960

        abar = torch.rand((1,1,num_users,hidden_dim), dtype=torch.bfloat16)
        cfg = ttnn.create_sharded_memory_config(shape=(1,1,num_users,hidden_dim), core_grid=ttnn.CoreGrid(y=num_users//32, x=8), strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=False)

        abar = ttnn.from_torch(abar, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=cfg)
        hidden_state = ttnn.to_memory_config(self.tt_hidden_state, cfg)
        output_tensor = ttnn.mul(abar, hidden_state, memory_config=cfg)

        self.tt_hidden_state = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        self.output = output_tensor
        return self.output
