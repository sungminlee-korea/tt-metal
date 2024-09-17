# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from torchvision import transforms, datasets
from loguru import logger

from torch.utils.data import DataLoader
from models.demos.functional_mnist.reference.mnist import MnistModel
from models.demos.functional_mnist.tt import ttnn_functional_mnist

from ttnn.model_preprocessing import preprocess_model_parameters


def run_demo_dataset(
    inputs_mesh_mapper,
    weights_mesh_mapper,
    output_mesh_composer,
    mesh_device,
    batch_size,
    iterations,
    model_location_generator,
):
    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Load model
    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=mesh_device,
    )
    correct = 0
    for iters in range(iterations):
        dataloader = DataLoader(test_dataset, batch_size=batch_size)
        x, labels = next(iter(dataloader))
        dataset_predictions = []
        ttnn_predictions = []
        dataset_ttnn_correct = 0
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper, device=mesh_device)
        tt_output = ttnn_functional_mnist.mnist(
            mesh_device, batch_size, x, parameters, mesh_mapper=inputs_mesh_mapper, mesh_composer=output_mesh_composer
        )
        tt_output = (
            ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer).permute(1, 2, 0, 3).squeeze(0).squeeze(0)
        )
        predicted_probabilities = torch.nn.functional.softmax(tt_output, dim=1)
        _, predicted_label = torch.max(predicted_probabilities, 1)
        tt_output = tt_output
        for i in range(batch_size):
            dataset_predictions.append(labels[i])
            ttnn_predictions.append(predicted_label[i])
            logger.info(f"Iter: {iters} Sample {i}:")
            logger.info(f"Expected Label: {dataset_predictions[i]}")
            logger.info(f"Predicted Label: {ttnn_predictions[i]}")

            if dataset_predictions[i] == ttnn_predictions[i]:
                dataset_ttnn_correct += 1
                correct += 1
        dataset_ttnn_accuracy = dataset_ttnn_correct / (batch_size)
        logger.info(
            f"ImageNet Inference Accuracy for iter {iters} of {batch_size} input samples : {dataset_ttnn_accuracy}"
        )

    accuracy = correct / (batch_size * iterations)
    logger.info(f"ImageNet Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("iterations", [1])
def test_demo_dataset(
    mesh_device,
    batch_size,
    iterations,
    model_location_generator,
):
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    return run_demo_dataset(
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
        mesh_device,
        batch_size=batch_size,
        iterations=iterations,
        model_location_generator=model_location_generator,
    )
