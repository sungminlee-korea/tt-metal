import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch

import tt_lib as ttl
from python_api_testing.models.utility_functions import nearest_32


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value",
    (
        ((1, 1, 3, 3), (1, 1, 5, 5), (0, 0, 1, 1), 0),
        ((1, 1, 3, 3), (2, 2, 5, 5), (0, 0, 0, 0), -1),
        ((1, 3, 30, 30), (1, 3, 32, 32), (0, 0, 0, 0), 1),
        ((1, 3, 30, 30), (3, 5, 32, 32), (1, 2, 0, 0), torch.inf),
        ((1, 3, 30, 30), (3, 3, 64, 64), (0, 0, 31, 31), -torch.inf),
    ),
)
def test_run_padding_test(
    input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value
):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)
    a_pt = torch.Tensor(a_pad.data()).reshape(*output_tensor_shape)

    # Pytorch reference
    input_tensor_end = tuple(
        input_tensor_start[i] + input_tensor_shape[i]
        for i in range(len(input_tensor_shape))
    )
    a_ref = torch.ones(*output_tensor_shape, dtype=torch.bfloat16) * pad_value
    a_ref[
        input_tensor_start[0] : input_tensor_end[0],
        input_tensor_start[1] : input_tensor_end[1],
        input_tensor_start[2] : input_tensor_end[2],
        input_tensor_start[3] : input_tensor_end[3],
    ] = inp

    print("\n", a_pt.shape)
    print("\n", a_pt)
    print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    assert torch.equal(a_pt, a_ref)


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_start, output_tensor_end",
    (
        ((1, 1, 5, 5), (0, 0, 1, 1), (0, 0, 3, 3)),
        ((2, 2, 5, 5), (0, 0, 0, 0), (0, 0, 2, 2)),
        ((1, 3, 32, 32), (0, 0, 0, 0), (0, 2, 29, 29)),
        ((3, 5, 32, 32), (1, 2, 0, 0), (1, 4, 29, 29)),
        ((3, 3, 64, 64), (0, 0, 32, 32), (0, 2, 61, 61)),
    ),
)
def test_run_unpadding_test(input_tensor_shape, output_tensor_start, output_tensor_end):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Unpad inputs on host
    output_tensor_shape = tuple(
        output_tensor_end[i] - output_tensor_start[i] + 1
        for i in range(len(input_tensor_shape))
    )
    a_unpad = a.unpad(output_tensor_start, output_tensor_end)
    a_pt = torch.Tensor(a_unpad.data()).reshape(*output_tensor_shape)

    # Pytorch reference
    a_ref = inp[
        output_tensor_start[0] : output_tensor_end[0] + 1,
        output_tensor_start[1] : output_tensor_end[1] + 1,
        output_tensor_start[2] : output_tensor_end[2] + 1,
        output_tensor_start[3] : output_tensor_end[3] + 1,
    ]

    print("\n", a_pt.shape)
    print("\n", a_pt)
    print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    assert torch.equal(a_pt, a_ref)


# Pad, run op, unpad
@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value",
    (((1, 1, 3, 4), (1, 1, 32, 32), (0, 0, 1, 1), 0),),
)
def test_run_padding_and_add_test(
    input_tensor_shape, output_tensor_shape, input_tensor_start, pad_value
):
    # Args for unpad
    output_tensor_start = input_tensor_start
    output_tensor_end = tuple(
        input_tensor_start[i] + input_tensor_shape[i] - 1
        for i in range(len(input_tensor_shape))
    )

    inp = torch.rand(*input_tensor_shape)
    ones = torch.ones(*input_tensor_shape)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    b = ttl.tensor.Tensor(
        ones.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)
    b_pad = b.pad(output_tensor_shape, input_tensor_start, pad_value)

    # Run add op on device with padded tensors
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_dev = a_pad.to(ttl.tensor.Layout.TILE).to(device)
    b_dev = b_pad.to(ttl.tensor.Layout.TILE).to(device)
    out_dev = ttl.tensor.add(a_dev, b_dev)
    out_pad = out_dev.to(host).to(ttl.tensor.Layout.ROW_MAJOR)

    # Unpad out to get result
    out = out_pad.unpad(output_tensor_start, output_tensor_end)
    out_pt = torch.Tensor(out.data()).reshape(*input_tensor_shape)

    out_ref = inp + ones

    print("\n", out_pt)
    print("\n", out_ref)

    assert torch.allclose(out_pt, out_ref, rtol=1e-2)

    del out_dev

    ttl.device.CloseDevice(device)


@pytest.mark.parametrize(
    "input_tensor_shape,  pad_value",
    (
        ((1, 1, 3, 3), 0),
        ((2, 2, 5, 5), -1),
        ((1, 3, 30, 30), 1),
        ((3, 5, 32, 32), torch.inf),
        ((3, 3, 66, 66), -torch.inf),
    ),
)
def test_run_tile_padding_test(input_tensor_shape, pad_value):
    output_tensor_shape = (
        *input_tensor_shape[:-2],
        nearest_32(input_tensor_shape[-2]),
        nearest_32(input_tensor_shape[-1]),
    )
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Pad inputs on host
    a_pad = a.pad_to_tile(pad_value)
    a_pt = torch.Tensor(a_pad.data()).reshape(output_tensor_shape)

    # Pytorch reference
    input_tensor_end = tuple(
        input_tensor_shape[i] for i in range(len(input_tensor_shape))
    )
    a_ref = torch.ones(*output_tensor_shape, dtype=torch.bfloat16) * pad_value
    a_ref[
        0 : input_tensor_end[0],
        0 : input_tensor_end[1],
        0 : input_tensor_end[2],
        0 : input_tensor_end[3],
    ] = inp

    print("\n", a_pt.shape)
    print("\n", a_pt)
    print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    assert torch.equal(a_pt, a_ref)


@pytest.mark.parametrize(
    "input_tensor_shape, output_tensor_shape",
    (
        ((1, 1, 32, 32), (1, 1, 4, 4)),
        ((2, 2, 32, 32), (2, 2, 32, 32)),
        ((1, 3, 64, 64), (1, 3, 33, 35)),
        ((3, 5, 32, 64), (3, 5, 31, 64)),
        ((3, 3, 64, 128), (3, 3, 64, 121)),
    ),
)
def test_run_tile_unpadding_test(input_tensor_shape, output_tensor_shape):
    inp = torch.rand(*input_tensor_shape, dtype=torch.bfloat16)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Unpad inputs on host
    a_unpad = a.unpad_from_tile(output_tensor_shape)
    a_pt = torch.Tensor(a_unpad.data()).reshape(*output_tensor_shape)

    # Pytorch reference
    a_ref = inp[
        0 : output_tensor_shape[0],
        0 : output_tensor_shape[1],
        0 : output_tensor_shape[2],
        0 : output_tensor_shape[3],
    ]

    print("\n", a_pt.shape)
    print("\n", a_pt)
    print("\n", a_ref)

    assert a_pt.shape == output_tensor_shape
    assert torch.equal(a_pt, a_ref)


# Pad, run op, unpad
@pytest.mark.parametrize(
    "input_tensor_shape, pad_value",
    (((1, 1, 3, 4), 0),),
)
def test_run_tile_padding_and_add_test(input_tensor_shape, pad_value):
    inp = torch.rand(*input_tensor_shape)
    ones = torch.ones(*input_tensor_shape)

    # Create tensor on host
    a = ttl.tensor.Tensor(
        inp.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    b = ttl.tensor.Tensor(
        ones.reshape(-1).tolist(),
        input_tensor_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Pad inputs on host
    a_pad = a.pad_to_tile(pad_value)
    b_pad = b.pad_to_tile(pad_value)

    # Run add op on device with padded tensors
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_dev = a_pad.to(ttl.tensor.Layout.TILE).to(device)
    b_dev = b_pad.to(ttl.tensor.Layout.TILE).to(device)
    out_dev = ttl.tensor.add(a_dev, b_dev)
    out_pad = out_dev.to(host).to(ttl.tensor.Layout.ROW_MAJOR)

    # Unpad out to get result
    out = out_pad.unpad_from_tile(input_tensor_shape)
    out_pt = torch.Tensor(out.data()).reshape(*input_tensor_shape)

    out_ref = inp + ones

    print("\n", out_pt)
    print("\n", out_ref)

    assert torch.allclose(out_pt, out_ref, rtol=1e-2)

    del out_dev

    ttl.device.CloseDevice(device)
