# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import BloomConfig, BloomForQuestionAnswering, BloomTokenizerFast

from models.experimental.functional_bloom.tt import ttnn_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import skip_for_wormhole_b0, comp_allclose_and_pcc

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_bloom])
def test_bloom_for_question_answering(device, use_program_cache, ttnn_model, batch_size=8, max_length=384):
    torch.manual_seed(0)

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForQuestionAnswering.from_pretrained(model_name).eval()

    num_heads = config.n_head

    question = "What is the main component of the aboral organ?"
    context = 'The largest single sensory feature is the aboral organ (at the opposite end from the mouth). Its main component is a statocyst, a balance sensor consisting of a statolith, a solid particle supported on four bundles of cilia, called "balancers", that sense its orientation. The statocyst is protected by a transparent dome made of long, immobile cilia. A ctenophore does not automatically try to keep the statolith resting equally on all the balancers. Instead its response is determined by the animal\'s "mood", in other words the overall state of the nervous system. For example, if a ctenophore with trailing tentacles captures prey, it will often put some comb rows into reverse, spinning the mouth towards the prey.'
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    num_tokens = inputs.input_ids.shape[-1]
    inputs.input_ids = inputs.input_ids.expand((batch_size, num_tokens))
    inputs.attention_mask = inputs.attention_mask.expand((batch_size, num_tokens))

    torch_output = torch_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    torch_start_logits = torch_output.start_logits
    torch_end_logits = torch_output.end_logits

    parameters = preprocess_model_parameters(
        model_name=f"ttnn_functional_bloom_for_question_answering",
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=ttnn_model.custom_preprocessor,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_ids, alibi, causal_mask = ttnn_model.preprocess_inputs(
        input_ids=input_ids, device=device, num_heads=num_heads, attention_mask=attention_mask, max_length=max_length
    )

    # Run twice to measure the time with and without the program cache
    tt_output = ttnn_model.bloom_for_question_answering(config, input_ids, alibi, causal_mask, parameters=parameters)

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)
    tt_start_logits = tt_output[:1, :num_tokens, 0]
    tt_end_logits = tt_output[:1, :num_tokens, 1]

    if ttnn_model == ttnn_functional_bloom:
        assert_with_pcc(torch_start_logits[0], tt_start_logits, 0.96677)
        assert_with_pcc(torch_end_logits[0], tt_end_logits, 0.95177)
    elif ttnn_model == ttnn_optimized_functional_bloom:
        print("start ", comp_allclose_and_pcc(torch_start_logits[:1], tt_start_logits, pcc=0.2))
        print("end ", comp_allclose_and_pcc(torch_end_logits[:1], tt_end_logits, pcc=0.2))
        assert_with_pcc(torch_start_logits[:1], tt_start_logits, 0.2)
        assert_with_pcc(torch_end_logits[:1], tt_end_logits, 0.2)
    else:
        raise RecursionError("Invalid ttnn_model")

    tt_start_logits_b1 = tt_output[0, :num_tokens, 0]
    tt_end_logits_b1 = tt_output[0, :num_tokens, 1]

    predict_answer_tokens_ttnn = inputs.input_ids[0, tt_start_logits_b1.argmax() : tt_end_logits_b1.argmax() + 1]
    ttnn_answer = tokenizer.decode(predict_answer_tokens_ttnn, skip_special_toekns=True)
    print("ttnn answer:", ttnn_answer)

    # assert torch_start_logits.argmax() == tt_start_logits.argmax()
    # assert torch_end_logits.argmax() == tt_end_logits.argmax()
