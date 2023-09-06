# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderModel

import tt_lib

from models.trocr.trocr_utils import GenerationMixin


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_causal_llm_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        generationmixin = GenerationMixin(
            model=model,
            config=model.decoder.config,
            state_dict=model.state_dict(),
            device=device,
        )
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
        pixel_values = processor(
            images=iam_ocr_sample_input, return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            input_ids = generationmixin.generate(pixel_values)

        generated_text = processor.batch_decode(input_ids, skip_special_tokens=True)[0]

        logger.info("TrOCR Model answered")
        logger.info(generated_text)
