from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
import ttnn
import pytest
from vgg_torch import VGG_TORCH
import torch

vgg_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def run_vgg11_inference(device, batch_size, num_classes, vgg_model_config):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    ###### ttnn 모델 만들기
    # torch-base 모델 파라미터 뽑기, 샘플 인풋 텐서 만들기
    torch_model = VGG_TORCH(num_classes=num_classes).eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=None,
    )

    input_shape = (batch_size, 3, 224, 224)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

    torch_model.to(torch.bfloat16)
    torch_input_tensor = torch_input_tensor.to(torch.bfloat16)
    # golden
    torch_output_tensor = torch_model(torch_input_tensor)
    breakpoint()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("num_classes", [1000])
def test_demo_sample(device, batch_size, num_classes):
    run_vgg11_inference(device, batch_size, num_classes)


if __name__ == "__main__":
    pass
