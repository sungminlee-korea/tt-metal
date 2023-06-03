from pathlib import Path
import sys
import torch
import torch.nn as nn
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
import tt_lib
from python_api_testing.models.roberta.roberta_self_attention import (
    TtRobertaSelfAttention,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from transformers import RobertaModel


def test_roberta_self_attention_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"encoder.layer.{SELF_ATTN_LAYER_INDEX}.attention.self"

    model = RobertaModel.from_pretrained("roberta-base")

    # Torch roberta self attn
    torch_model = model.encoder.layer[SELF_ATTN_LAYER_INDEX].attention.self

    # Tt roberta self attn
    tt_model = TtRobertaSelfAttention(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )
    # Run torch model
    hidden_states = torch.rand(1, 32, 768)
    attention_mask = torch.ones(1, 1, 32)
    torch_output = torch_model(hidden_states, attention_mask=attention_mask)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch2tt_tensor(hidden_states, device)
    attention_mask = torch.unsqueeze(attention_mask, 0)
    tt_attention_mask = torch2tt_tensor(attention_mask, device)

    tt_output = tt_model(tt_hidden_states, attention_mask=tt_attention_mask)

    tt_output_torch = tt2torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("RobertaSelfAttention Passed!")
    else:
        logger.warning("RobertaSelfAttention Failed!")

    assert does_pass


if __name__ == "__main__":
    test_roberta_self_attention_inference()
