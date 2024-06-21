# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [0.05, 1.0, 0.5, 0.12])
def test_bw_addalpha(input_shapes, alpha, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.addalpha_bw(grad_tensor, input_tensor, other_tensor, alpha)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.add(in_data, other_data, alpha=alpha)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [0.05, 2.0, 1.5, 0.12])
# @pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
@pytest.mark.parametrize("are_required_outputs", [[True, True]])
def test_bw_addalpha_with_opt_output(input_shapes, alpha, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    input_grad = None
    other_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.addalpha_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        alpha,
        are_required_outputs=are_required_outputs,
        optional_input_a_grad=input_grad,
        optional_input_b_grad=other_grad,
        queue_id=cq_id,
    )

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.add(in_data, other_data, alpha=alpha)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status
