# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time
from models.utility_functions import skip_for_grayskull
from models.experimental.yolov4.reference.downsample5 import DownSample5
from models.experimental.yolov4.ttnn_opt_cb.downsample5 import Down5
from loguru import logger


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_down5(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")

    if model_path == "models":
        pytest.skip(
            "Requires weights file to be downloaded from https://drive.google.com/file/d/1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ/view"
        )
    else:
        weights_pth = str(model_path / "yolov4.pth")

    ttnn_model = Down5(weights_pth)

    torch_input = torch.randn((1, 20, 20, 512), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_model = DownSample5()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items() if (k.startswith("down5."))}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_ttnn = ttnn_model(device, ttnn_input)

    result = ttnn.to_torch(result_ttnn)
    ref = torch_model(torch_input)
    ref = ref.permute(0, 2, 3, 1)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.93)  # PCC 0.93
