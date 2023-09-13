# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import random
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.models.roberta.roberta_for_question_answering import (
    TtRobertaForQuestionAnswering,
)

import tt_lib
from tt_lib.fallback_ops import fallback_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from transformers import RobertaForQuestionAnswering
from transformers import AutoTokenizer


def test_roberta_qa_inference(device):
    torch.manual_seed(1234)

    base_address = f""

    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    torch_model = RobertaForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2"
    )
    torch_model.eval()

    tt_model = TtRobertaForQuestionAnswering(
        config=torch_model.config,
        base_address=base_address,
        device=device,
        state_dict=torch_model.state_dict(),
        reference_model=torch_model,
    )
    tt_model.eval()

    with torch.no_grad():
        question, context = (
            "Where do I live?",
            "My name is Merve and I live in İstanbul.",
        )

        inputs = tokenizer(question, context, return_tensors="pt")

        torch_output = torch_model(**inputs)

        torch_answer_start_index = torch_output.start_logits.argmax()
        torch_answer_end_index = torch_output.end_logits.argmax()

        torch_predict_answer_tokens = inputs.input_ids[
            0, torch_answer_start_index : torch_answer_end_index + 1
        ]
        torch_answer = tokenizer.decode(
            torch_predict_answer_tokens, skip_special_tokens=True
        )
        logger.info("Torch answered")
        logger.info(torch_answer)

        # TT
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

        tt_output = tt_model(inputs.input_ids, tt_attention_mask)

        tt_answer_start_index = tt_output.start_logits.argmax()
        tt_answer_end_index = tt_output.end_logits.argmax()

        tt_predict_answer_tokens = inputs.input_ids[
            0, tt_answer_start_index : tt_answer_end_index + 1
        ]
        tt_answer = tokenizer.decode(tt_predict_answer_tokens, skip_special_tokens=True)
        logger.info("TT answered")
        logger.info(tt_answer)

        # Compare outputs
        does_pass_1, pcc_message = comp_pcc(
            torch_output.start_logits, tt_output.start_logits, 0.98
        )

        logger.info(comp_allclose(torch_output.start_logits, tt_output.start_logits))
        logger.info(pcc_message)

        does_pass_2, pcc_message = comp_pcc(
            torch_output.end_logits, tt_output.end_logits, 0.98
        )

        logger.info(comp_allclose(torch_output.end_logits, tt_output.end_logits))
        logger.info(pcc_message)


        if does_pass_1 and does_pass_2:
            logger.info("RobertaForQA Passed!")
        else:
            logger.warning("RobertaForQA Failed!")

        assert does_pass_1 and does_pass_2
