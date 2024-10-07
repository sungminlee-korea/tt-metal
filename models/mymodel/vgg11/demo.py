from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
from loguru import logger
import ttnn
import pytest
from .vgg_torch import VGG_TORCH
from .vgg_ttnn import VGG_TTNN
import torch

vgg_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def run_vgg11_inference(device, batch_size, num_classes, vgg_model_config):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    # 0. config 만들기
    model_config = {
        "MATH_FIDELITY": vgg_model_config["MATH_FIDELITY"],
        "WEIGHTS_DTYPE": vgg_model_config["WEIGHTS_DTYPE"],
        "ACTIVATIONS_DTYPE": vgg_model_config["ACTIVATIONS_DTYPE"],
    }

    # 1. torch-base 모델 파라미터 뽑기, 샘플 인풋 텐서 만들기, golden ans
    torch_model = VGG_TORCH(num_classes=num_classes).eval()
    parameters = torch_model.state_dict()
    input_shape = (batch_size, 3, 224, 224)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    # torch_model.to(torch.bfloat16)
    # golden
    with torch.no_grad():
        torch_output_tensor = torch_model(torch_input_tensor)

    ttnn_model = VGG_TTNN(
        device=device,
        torch_model=torch_model,
        torch_input=torch_input_tensor,
        parameters=parameters,
        batch_size=batch_size,
        model_config=vgg_model_config,
        input_shape=input_shape,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        num_classes=num_classes,
        dealloc_input=True,
    )
    ttnn.synchronize_device(device)
    # 2. ttnn tensor 변환
    tt_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), ttnn.bfloat16)
    ttnn.synchronize_device(device)

    # 3. run
    tt_output_tensor = ttnn_model(tt_input_tensor)
    ttnn.synchronize_device(device)
    tt_output_tensor = ttnn.from_device(tt_output_tensor).to_torch().reshape((batch_size, num_classes))
    pcc = 0.9
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, tt_output_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_classes", [1000])
def test_demo_sample(device, batch_size, num_classes):
    run_vgg11_inference(device, batch_size, num_classes, vgg_model_config)


if __name__ == "__main__":
    pass
    """
    pytest models/mymodel/vgg11/demo.py::test_demo_sample
    """
