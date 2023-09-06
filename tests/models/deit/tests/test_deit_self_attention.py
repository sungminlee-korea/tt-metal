# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from loguru import logger

import tt_lib
from models.utility_functions import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig

from transformers import DeiTModel
from deit_self_attention import TtDeiTSelfAttention


def test_deit_self_attention_inference(pcc = 0.99):

    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'encoder.layer.0.attention.attention'
    torch_self_attention = model.encoder.layer[0].attention.attention
    head_mask = None
    output_attentions = False
    input_shape =  torch.Size([1, 1, 198, 768])
    hidden_state = torch.randn(input_shape)

    torch_output = torch_self_attention(hidden_state.squeeze(0), head_mask, output_attentions)[0]

    # Initialize the device
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    # setup tt model
    tt_self_attention = TtDeiTSelfAttention(DeiTConfig(), device, state_dict, base_address)

    tt_input = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_out = tt_self_attention(tt_input, head_mask, output_attentions)
    tt_output = tt_to_torch_tensor(tt_out[0]).squeeze(0)

    passing = comp_pcc(torch_output, tt_output, pcc)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output, pcc))
    tt_lib.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
