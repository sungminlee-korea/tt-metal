# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_dwconv import TtSegformerDWConv

program_configs = {
    "linear_config_4096": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=2,
        out_subblock_w=8,
        per_core_M=2,
        per_core_N=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
}


class TtSegformerMixFFN:
    def __init__(self, parameters, hidden_features, model):
        super().__init__()
        self.dwconv = TtSegformerDWConv(parameters["dwconv"], hidden_features, model.dwconv)

    def __call__(self, hidden_states: ttnn.Tensor, height: int, width: int, parameters, device):
        print("mm-7--", hidden_states.shape)
        # hidden_states = ttnn.linear(
        #     hidden_states,
        #     parameters.dense1.weight,
        #     bias=parameters.dense1.bias,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     core_grid=ttnn.CoreGrid(y=8, x=8),
        #     dtype=ttnn.bfloat8_b,
        # )

        # print("mm-1--", hidden_states.shape)
        if hidden_states.shape[1] > 4096:
            mm_78_y = 8
            mm_78_x = 4
        if hidden_states.shape[1] == 4096:
            mm_78_y = 8
            mm_78_x = 8
        elif hidden_states.shape[1] == 1024:
            mm_78_y = 8
            mm_78_x = 5
        elif hidden_states.shape[1] == 256:
            mm_78_y = 8
            mm_78_x = 4

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        if hidden_states.shape[1] > 4096:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense1.weight,
                bias=parameters.dense1.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 4096:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense1.weight,
                bias=parameters.dense1.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                program_config=program_configs["linear_config_4096"],
            )
        elif hidden_states.shape[1] == 1024:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense1.weight,
                bias=parameters.dense1.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 256:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense1.weight,
                bias=parameters.dense1.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                dtype=ttnn.bfloat8_b,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        hidden_states = self.dwconv(hidden_states, height, width, device)
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        print("mm-8--", hidden_states.shape)
        # hidden_states = ttnn.linear(
        #     hidden_states,
        #     parameters.dense2.weight,
        #     bias=parameters.dense2.bias,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     core_grid=ttnn.CoreGrid(y=8, x=8),
        #     dtype=ttnn.bfloat16,
        # )
        if hidden_states.shape[1] > 4096:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense2.weight,
                bias=parameters.dense2.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 4096:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense2.weight,
                bias=parameters.dense2.bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
            )
        elif hidden_states.shape[1] == 1024:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense2.weight,
                bias=parameters.dense2.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                dtype=ttnn.bfloat8_b,
            )
        elif hidden_states.shape[1] == 256:
            hidden_states = ttnn.to_memory_config(
                hidden_states,
                memory_config=ttnn.create_sharded_memory_config(
                    hidden_states.shape,
                    core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            hidden_states = ttnn.linear(
                hidden_states,
                parameters.dense2.weight,
                bias=parameters.dense2.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(y=mm_78_y, x=mm_78_x),
                dtype=ttnn.bfloat8_b,
            )
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return hidden_states
