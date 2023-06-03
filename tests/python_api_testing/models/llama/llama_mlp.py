import torch
from torch import nn
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.llama.llama_utils import (
    tt2torch_tensor,
    torch2tt_tensor,
    linear,
)


class TtLlamaMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = None

        self.out_gate_proj = torch2tt_tensor(
            self.state_dict[f"{base_url}.{layer_num}.mlp.gate_proj.weight"], self.device
        )
        self.out_down_proj = torch2tt_tensor(
            self.state_dict[f"{base_url}.{layer_num}.mlp.down_proj.weight"], self.device
        )
        self.out_up_proj = torch2tt_tensor(
            self.state_dict[f"{base_url}.{layer_num}.mlp.up_proj.weight"], self.device
        )

        if hidden_act == "silu":  # silu
            self.act_fn = fallback_ops.silu

    def forward(self, x):
        # gate proj
        gate = linear(x, self.out_gate_proj, self.bias)
        # apply silu activation function
        gate = self.act_fn(gate)

        # up proj
        up = linear(x, self.out_up_proj, self.bias)

        # product
        prod = tt_lib.tensor.mul(gate, up)

        # down
        hidden_states = linear(prod, self.out_down_proj, self.bias)

        # return TT Tensor
        return hidden_states
