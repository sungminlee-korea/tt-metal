from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torchvision import models
from loguru import logger
from PIL import Image
import pytest

from libs import tt_lib
from utility_functions import comp_pcc
from vgg import *


_batch_size = 1


def run_vgg_inference(image_path, pcc):
    im = Image.open(image_path)
    im = im.resize((224, 224))

    # Apply the transformation to the random image and Add an extra dimension at the beginning
    # to match the desired shape of 3x224x224
    image = transforms.ToTensor()(im).unsqueeze(0)

    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        torch_vgg.eval()

        state_dict = torch_vgg.state_dict()
        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device, host, state_dict, disable_conv_on_tt_device=True)

        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_vgg(image)

        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert(
            pcc_passing
        ), f"Model output does not meet PCC requirement {pcc}."

@pytest.mark.parametrize(
    "path_to_image, pcc",
    (
        ("sample_image.JPEG", 0.99),
    ),
)
def test_vgg_inference(path_to_image, pcc):
    run_vgg_inference(path_to_image, pcc)

if __name__ == "__main__":
    run_vgg_inference("sample_image.JPEG", 0.99)
