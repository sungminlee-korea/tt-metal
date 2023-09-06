# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
import torch
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)

import tt_lib
from tests.models.swin.tt.swin_for_image_classification import (
    TtSwinForImageClassification,
)
from tests.models.swin.swin_utils import get_shape
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from transformers import AutoFeatureExtractor


@pytest.mark.parametrize(
    "model_name, pcc",
    (("microsoft/swin-tiny-patch4-window7-224", 0.99),),
)
def test_swin_image_classification_inference(imagenet_sample_input, model_name, pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HF_SwinForImageClassification.from_pretrained(model_name)

    image = imagenet_sample_input
    inputs = feature_extractor(images=image, return_tensors="pt")

    base_address = f"swin."

    with torch.no_grad():
        torch_model = model

        tt_model = TtSwinForImageClassification(
            config=model.config,
            state_dict=model.state_dict(),
            base_address=base_address,
            device=device,
            host=host,
        )

        # Run torch model
        torch_output = torch_model(**inputs)

        # Run tt model
        tt_pixel_values = inputs.pixel_values
        tt_pixel_values = torch_to_tt_tensor_rm(tt_pixel_values, device)

        tt_output = tt_model(tt_pixel_values)

        tt_output_torch = tt_to_torch_tensor(tt_output.logits)
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)
        does_pass, pcc_message = comp_pcc(torch_output.logits, tt_output_torch, pcc)

        logger.info(comp_allclose(torch_output.logits, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)

        if does_pass:
            logger.info("SwinForImageClassification Passed!")
        else:
            logger.warning("SwinForImageClassification Failed!")

        assert does_pass
